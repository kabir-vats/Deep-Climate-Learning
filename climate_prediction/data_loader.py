from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import random
import math
import numpy as np
import xarray as xr
import dask.array as da
import lightning.pytorch as pl
import os
from global_land_mask import globe
from climate_prediction.util import get_lat_weights, Normalizer


class ClimateSequenceDataset(Dataset):
    def __init__(self, input_list, output_list, sequence_length=12):
        self.inputs, self.outputs = [], []

        for i in range(len(input_list)):
            inputs_dask = input_list[i]
            outputs_dask = output_list[i]

            inputs_np = inputs_dask.compute()
            outputs_np = outputs_dask.compute()
            self.sequence_length = sequence_length
            for j in range(len(inputs_np) - sequence_length):
                x_seq = inputs_np[j : j + sequence_length]  # [seq_len, C, H, W]
                y = outputs_np[j + sequence_length - 1]  # [2, H, W]
                self.inputs.append(torch.from_numpy(x_seq).float())
                self.outputs.append(torch.from_numpy(y).float())
        self.inputs = torch.stack(self.inputs)
        self.outputs = torch.stack(self.outputs)

        if torch.isnan(self.inputs).any() or torch.isnan(self.outputs).any():
            raise ValueError("NaNs found in dataset")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class ClimateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        input_vars,
        output_vars,
        train_ssps,
        test_ssp,
        target_member_id,
        train_member_ids,
        val_member_id,
        val_split=0.1,
        test_months=120,
        val_months=120,
        batch_size=32,
        num_workers=0,
        sequence_length=12,
        seed=42,
    ):
        super().__init__()
        self.path = path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.train_ssps = train_ssps
        self.test_ssp = test_ssp
        self.target_member_id = target_member_id
        self.val_split = val_split
        self.test_months = test_months
        self.val_months = val_months
        self.train_member_ids = train_member_ids
        self.val_member_id = val_member_id
        self.test_member_id = target_member_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.seed = seed
        self.normalizer = Normalizer()

    def prepare_data(self):
        assert os.path.exists(self.path), f"Data path not found: {self.path}"

    def setup(self, stage=None):
        ds = xr.open_zarr(self.path, consolidated=False, chunks={"time": 24})
        spatial_template = ds["rsdt"].isel(time=0, ssp=0, drop=True)

        def make_rolling_features(da_var, windows=[3, 12, 60]):
            """
            For each window in months, compute the rolling mean
            with min_periods=1 so the first entries don’t become NaN
            (you just get the mean over whatever history you have).
            Returns a list of DataArrays.
            """
            rolls = []
            for w in windows:
                # compute rolling mean over time; the first (w-1) entries will be
                # mean over [0:idx], not NaN
                r = da_var.rolling(time=w, min_periods=1).mean()
                # optionally give a sensible name for debugging
                r = r.rename(f"{da_var.name}_roll{w}mo")
                rolls.append(r)
            return rolls

        def get_sin_cos_month():
            month_index = ds.time.dt.month - 1
            month_phase = 2 * np.pi * month_index / 12
            sin_month = (
                xr.DataArray(np.sin(month_phase), dims=["time"])
                .broadcast_like(spatial_template)
                .transpose("time", "y", "x")
                .data
            )
            cos_month = (
                xr.DataArray(np.cos(month_phase), dims=["time"])
                .broadcast_like(spatial_template)
                .transpose("time", "y", "x")
                .data
            )
            return sin_month, cos_month

        """
        def get_sin_cos_month():
            month_index = ds.time.dt.month - 1
            month_phase = 2 * np.pi * month_index / 12
            sin_month = xr.DataArray(np.sin(month_phase), dims=["time"]).data
            cos_month = xr.DataArray(np.cos(month_phase), dims=["time"]).data
            return sin_month, cos_month
        """

        def get_sin_cos_lat():
            lat_vals = np.deg2rad(spatial_template.y.values)
            lat_sin = (
                xr.DataArray(np.sin(lat_vals), dims=["y"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            lat_cos = (
                xr.DataArray(np.cos(lat_vals), dims=["y"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            return lat_sin, lat_cos

        def get_sin_cos_lon():
            lon_vals = np.deg2rad(spatial_template.x.values)

            lon_sin = (
                xr.DataArray(np.sin(lon_vals), dims=["x"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            lon_cos = (
                xr.DataArray(np.cos(lon_vals), dims=["x"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            return lon_sin, lon_cos

        def get_landmask():
            lat2d, lon2d = np.meshgrid(
                spatial_template.y.values, spatial_template.x.values, indexing="ij"
            )
            lon2d_wrapped = ((lon2d + 180) % 360) - 180
            landmask_np = globe.is_land(lat2d, lon2d_wrapped).astype(np.float32)
            landmask_da = (
                xr.DataArray(landmask_np, dims=("y", "x"))
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
            )
            return landmask_da

        def get_year():
            year_index = ds.time.dt.year - 1
            year = (
                xr.DataArray(year_index, dims=["time"])
                .broadcast_like(spatial_template)
                .transpose("time", "y", "x")
                .data
            )
            return year

        def load_ssp(ssp, member_ids):
            n_members = len(member_ids)

            def _collect(var, is_output=False):
                da_var = ds[var].sel(ssp=ssp)
                if "latitude" in da_var.dims:
                    da_var = da_var.rename({"latitude": "y", "longitude": "x"})
                # list-wise processing keeps chronological order inside each member,
                # then concatenates along time axis
                tensors = []
                for m in member_ids:
                    slice_m = (
                        da_var.sel(member_id=m)
                        if "member_id" in da_var.dims
                        else da_var
                    )
                    if set(slice_m.dims) == {"time"}:
                        slice_m = slice_m.broadcast_like(spatial_template).transpose(
                            "time", "y", "x"
                        )
                    tensors.append(slice_m.data)
                return da.concatenate(tensors, axis=0)  # axis 0 == time

            input_dask, output_dask = [], []
            for var in self.input_vars:
                input_dask.append(_collect(var, is_output=False))

            sin_month, cos_month = get_sin_cos_month()
            sin_lat, cos_lat = get_sin_cos_lat()
            sin_lon, cos_lon = get_sin_cos_lon()
            landmask = get_landmask().data
            # year = get_year()

            def _rep(a):
                return da.concatenate([a] * n_members, axis=0) if n_members > 1 else a

            input_dask.extend(
                [
                    _rep(sin_month),
                    _rep(cos_month),
                    _rep(sin_lat),
                    _rep(cos_lat),
                    _rep(sin_lon),
                    _rep(cos_lon),
                    _rep(landmask),
                ]
            )

            for var in self.output_vars:
                output_dask.append(_collect(var, is_output=True))
            return da.stack(input_dask, axis=1), da.stack(output_dask, axis=1)

        train_input, train_output, val_input, val_output = [], [], None, None

        for ssp in self.train_ssps:
            x, y = load_ssp(ssp, self.train_member_ids)
            if ssp == "ssp370":
                n_members = len(self.train_member_ids)
                n_time = ds.sizes["time"]  # months per member
                val_len = self.val_months  # no +sequence_length
                total_time = n_members * n_time

                # drop last val_len months for every member
                train_mask = np.ones(total_time, dtype=bool)
                for i in range(n_members):
                    start = i * n_time + (n_time - val_len)
                    train_mask[start : (i + 1) * n_time] = False

                # validation slice for the chosen member, with lead-up window
                x_val, y_val = load_ssp(ssp, [self.val_member_id])  # single member
                v_start = n_time - (val_len + self.sequence_length)
                val_input = x_val[v_start:]
                val_output = y_val[v_start:]

                train_input.append(x[train_mask])
                train_output.append(y[train_mask])
            else:
                train_input.append(x)
                train_output.append(y)

        train_input_combined = da.concatenate(train_input, axis=0)
        train_output_combined = da.concatenate(train_output, axis=0)

        self.normalizer.set_input_statistics(
            mean=da.nanmean(
                train_input_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
            std=da.nanstd(
                train_input_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
        )
        self.normalizer.set_output_statistics(
            mean=da.nanmean(
                train_output_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
            std=da.nanstd(
                train_output_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
        )

        train_input_norm = [
            self.normalizer.normalize(ssp_input, "input") for ssp_input in train_input
        ]
        train_output_norm = [
            self.normalizer.normalize(ssp_output, "output")
            for ssp_output in train_output
        ]
        val_input_norm = self.normalizer.normalize(val_input, "input")
        val_output_norm = self.normalizer.normalize(val_output, "output")

        test_input, test_output = load_ssp(self.test_ssp, [self.test_member_id])
        test_input = test_input[-self.test_months - self.sequence_length :]
        test_output = test_output[-self.test_months - self.sequence_length :]
        test_input_norm = self.normalizer.normalize(test_input, "input")

        self.train_dataset = ClimateSequenceDataset(
            train_input_norm, train_output_norm, sequence_length=self.sequence_length
        )
        self.val_dataset = ClimateSequenceDataset(
            [val_input_norm], [val_output_norm], sequence_length=self.sequence_length
        )
        self.test_dataset = ClimateSequenceDataset(
            [test_input_norm],
            [test_output],
            sequence_length=self.sequence_length,
        )

        self.lat = spatial_template.y.values
        self.lon = spatial_template.x.values
        self.area_weights = xr.DataArray(
            get_lat_weights(self.lat), dims=["y"], coords={"y": self.lat}
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_lat_weights(self):
        return self.area_weights

    def get_coords(self):
        return self.lat, self.lon


class ClimateDecadeDataset(Dataset):
    """
    For every sample we also store a metadata tuple (ssp, year)
    so that custom samplers can group by decade, SSP, etc.
    """

    def __init__(
        self,
        input_list,  # list of Dask arrays  [T, C, H, W]
        output_list,  # same length
        ssp_labels,  # list[str]           ← NEW
        time_lists,  # list[np.ndarray]    ← NEW, same len
        sequence_length=12,
    ):
        self.inputs, self.outputs, self.metadata = [], [], []
        self.sequence_length = sequence_length

        for block, ssp, times in zip(range(len(input_list)), ssp_labels, time_lists):

            x_np = input_list[block].compute()
            y_np = output_list[block].compute()

            # --- main sliding-window loop ------------------------------
            for j in range(len(x_np) - sequence_length):
                x_seq = x_np[j : j + sequence_length]  # [seq, C, H, W]
                y = y_np[j + sequence_length - 1]  # [V, H, W]

                self.inputs.append(torch.from_numpy(x_seq).float())
                self.outputs.append(torch.from_numpy(y).float())

                # ----------  metadata  --------------------------------
                # year of the *target* month
                target_ts = times[j + sequence_length - 1]  # np.datetime64
                year = target_ts.year
                self.metadata.append((ssp[j + sequence_length - 1], year))
        # --------------------------------------------------------------

        self.inputs = torch.stack(self.inputs)
        self.outputs = torch.stack(self.outputs)

        if torch.isnan(self.inputs).any() or torch.isnan(self.outputs).any():
            raise ValueError("NaNs found in dataset")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class DecadeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._build_groups()  # collect indices by (ssp, decade)

    def _build_groups(self):
        self.groups = defaultdict(list)
        for idx, (ssp, year) in enumerate(self.dataset.metadata):
            self.groups[(ssp, (year // 10) * 10)].append(idx)

    def _make_batches(self, g):
        batches = []
        for idxs in self.groups.values():
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g)
                idxs = [idxs[i] for i in perm]
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i : i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        # rebuild & reshuffle **every** epoch
        g = torch.Generator()
        g.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item() ^ self.seed)
        return iter(self._make_batches(g))

    def __len__(self):
        # number of batches can change if we reshuffle tails differently
        return sum(math.ceil(len(v) / self.batch_size) for v in self.groups.values())


class ClimateDecadeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        input_vars,
        output_vars,
        train_ssps,
        test_ssp,
        target_member_id,
        train_member_ids,
        val_member_id,
        val_split=0.1,
        test_months=120,
        val_months=120,
        batch_size=32,
        num_workers=0,
        sequence_length=12,
        seed=42,
    ):
        super().__init__()
        self.path = path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.train_ssps = train_ssps
        self.test_ssp = test_ssp
        self.test_member_id = target_member_id
        self.train_member_ids = train_member_ids
        self.val_member_id = val_member_id
        self.val_split = val_split
        self.test_months = test_months
        self.val_months = val_months
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.seed = seed
        self.normalizer = Normalizer()

    def prepare_data(self):
        assert os.path.exists(self.path), f"Data path not found: {self.path}"

    def setup(self, stage=None):
        ds = xr.open_zarr(self.path, consolidated=False, chunks={"time": 24})
        spatial_template = ds["rsdt"].isel(time=0, ssp=0, drop=True)

        def get_sin_cos_month():
            month_index = ds.time.dt.month - 1
            month_phase = 2 * np.pi * month_index / 12
            sin_month = (
                xr.DataArray(np.sin(month_phase), dims=["time"])
                .broadcast_like(spatial_template)
                .transpose("time", "y", "x")
                .data
            )
            cos_month = (
                xr.DataArray(np.cos(month_phase), dims=["time"])
                .broadcast_like(spatial_template)
                .transpose("time", "y", "x")
                .data
            )
            return sin_month, cos_month

        def get_sin_cos_lat():
            lat_vals = np.deg2rad(spatial_template.y.values)
            lat_sin = (
                xr.DataArray(np.sin(lat_vals), dims=["y"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            lat_cos = (
                xr.DataArray(np.cos(lat_vals), dims=["y"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            return lat_sin, lat_cos

        def get_sin_cos_lon():
            lon_vals = np.deg2rad(spatial_template.x.values)

            lon_sin = (
                xr.DataArray(np.sin(lon_vals), dims=["x"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            lon_cos = (
                xr.DataArray(np.cos(lon_vals), dims=["x"])
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
                .data
            )
            return lon_sin, lon_cos

        def get_landmask():
            lat2d, lon2d = np.meshgrid(
                spatial_template.y.values, spatial_template.x.values, indexing="ij"
            )
            lon2d_wrapped = ((lon2d + 180) % 360) - 180
            landmask_np = globe.is_land(lat2d, lon2d_wrapped).astype(np.float32)
            landmask_da = (
                xr.DataArray(landmask_np, dims=("y", "x"))
                .broadcast_like(spatial_template)
                .expand_dims(time=ds.sizes["time"])
            )
            return landmask_da

        def load_ssp(ssp, member_ids):
            n_members = len(member_ids)

            def _collect(var, is_output=False):
                da_var = ds[var].sel(ssp=ssp)
                if "latitude" in da_var.dims:
                    da_var = da_var.rename({"latitude": "y", "longitude": "x"})
                # list-wise processing keeps chronological order inside each member,
                # then concatenates along time axis
                tensors = []
                for m in member_ids:
                    slice_m = (
                        da_var.sel(member_id=m)
                        if "member_id" in da_var.dims
                        else da_var
                    )
                    if set(slice_m.dims) == {"time"}:
                        slice_m = slice_m.broadcast_like(spatial_template).transpose(
                            "time", "y", "x"
                        )
                    tensors.append(slice_m.data)
                return da.concatenate(tensors, axis=0)  # axis 0 == time

            input_dask, output_dask = [], []
            for var in self.input_vars:
                input_dask.append(_collect(var, is_output=False))

            sin_month, cos_month = get_sin_cos_month()
            sin_lat, cos_lat = get_sin_cos_lat()
            sin_lon, cos_lon = get_sin_cos_lon()
            landmask = get_landmask().data

            def _rep(a):
                return da.concatenate([a] * n_members, axis=0) if n_members > 1 else a

            input_dask.extend(
                [
                    _rep(sin_month),
                    _rep(cos_month),
                    _rep(sin_lat),
                    _rep(cos_lat),
                    _rep(sin_lon),
                    _rep(cos_lon),
                    _rep(landmask),
                ]
            )

            for var in self.output_vars:
                output_dask.append(_collect(var, is_output=True))
            return da.stack(input_dask, axis=1), da.stack(output_dask, axis=1)

        train_input, train_output, val_input, val_output = [], [], None, None

        train_times, train_ssps, val_times, val_ssps = (
            [],
            [],
            [],
            [],
        )

        for ssp in self.train_ssps:
            x, y = load_ssp(ssp, self.train_member_ids)
            times = ds.time.values
            if ssp == "ssp370":

                val_times.append(times[-self.val_months - self.sequence_length :])
                val_ssps.append([ssp] * len(val_times[-1]))

                n_members = len(self.train_member_ids)
                n_time = ds.sizes["time"]  # months per member
                val_len = self.val_months  # no +sequence_length
                total_time = n_members * n_time

                # drop last val_len months for every member
                train_mask = np.ones(total_time, dtype=bool)
                for i in range(n_members):
                    start = i * n_time + (n_time - val_len)
                    train_mask[start : (i + 1) * n_time] = False

                # validation slice for the chosen member, with lead-up window
                x_val, y_val = load_ssp(ssp, [self.val_member_id])  # single member
                v_start = n_time - (val_len + self.sequence_length)
                val_input = x_val[v_start:]
                val_output = y_val[v_start:]

                train_times.append(
                    np.array(
                        list(times[: -self.val_months]) * len(self.train_member_ids)
                    )
                )
                train_ssps.append([ssp] * len(train_times[-1]))
                train_input.append(x[train_mask])
                train_output.append(y[train_mask])
            else:
                train_input.append(x)
                train_output.append(y)
                train_times.append(np.array(list(times) * len(self.train_member_ids)))
                train_ssps.append([ssp] * len(train_times[-1]))

        train_input_combined = da.concatenate(train_input, axis=0)
        train_output_combined = da.concatenate(train_output, axis=0)

        self.normalizer.set_input_statistics(
            mean=da.nanmean(
                train_input_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
            std=da.nanstd(
                train_input_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
        )
        self.normalizer.set_output_statistics(
            mean=da.nanmean(
                train_output_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
            std=da.nanstd(
                train_output_combined, axis=(0, 2, 3), keepdims=True
            ).compute(),
        )

        train_input_norm = [
            self.normalizer.normalize(ssp_input, "input") for ssp_input in train_input
        ]
        train_output_norm = [
            self.normalizer.normalize(ssp_output, "output")
            for ssp_output in train_output
        ]
        val_input_norm = self.normalizer.normalize(val_input, "input")
        val_output_norm = self.normalizer.normalize(val_output, "output")

        test_input, test_output = load_ssp(self.test_ssp, [self.test_member_id])
        times = ds.time.values
        test_times, test_ssps = [], []
        test_times.append(times[-self.test_months - self.sequence_length :])
        test_ssps.append([ssp] * len(test_times[-1]))
        test_input = test_input[-self.test_months - self.sequence_length :]

        test_output = test_output[-self.test_months - self.sequence_length :]
        test_input_norm = self.normalizer.normalize(test_input, "input")

        self.train_dataset = ClimateDecadeDataset(
            train_input_norm,
            train_output_norm,
            ssp_labels=train_ssps,
            time_lists=train_times,
            sequence_length=self.sequence_length,
        )
        self.val_dataset = ClimateDecadeDataset(
            [val_input_norm],
            [val_output_norm],
            ssp_labels=val_ssps,
            time_lists=val_times,
            sequence_length=self.sequence_length,
        )
        self.test_dataset = ClimateDecadeDataset(
            [test_input_norm],
            [test_output],
            ssp_labels=test_ssps,
            time_lists=test_times,
            sequence_length=self.sequence_length,
        )

        self.lat = spatial_template.y.values
        self.lon = spatial_template.x.values
        self.area_weights = xr.DataArray(
            get_lat_weights(self.lat), dims=["y"], coords={"y": self.lat}
        )

    def train_dataloader(self):
        sampler = DecadeSampler(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, seed=self.seed
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = DecadeSampler(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, seed=self.seed
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        sampler = DecadeSampler(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, seed=self.seed
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_lat_weights(self):
        return self.area_weights

    def get_coords(self):
        return self.lat, self.lon
