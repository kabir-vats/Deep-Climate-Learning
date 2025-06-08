import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from climate_prediction.loss import AreaWeightedClimateLoss, L1CustomLoss, MSECustomLoss
import os
import xarray as xr
import datetime


# FINE TUNING VERSION
class ClimateEmulationModule(pl.LightningModule):
    def __init__(self, model, output_vars, loss_config, optimizer_config):
        super().__init__()
        self.model = model
        self.save_hyperparameters(
            ignore=["model"]
        )  # Save all hyperparameters except the model to self.hparams.<param_name>
        self.optimizer_config = optimizer_config

        # Decide Loss
        loss_version = loss_config["version"]
        if loss_version == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_version == "L1":
            self.criterion = nn.L1Loss()
        elif loss_version == "Custom":
            self.criterion = AreaWeightedClimateLoss(
                var_names=output_vars,
                train_weights=loss_config.get("train_weights", None),
            )
        elif loss_version == "L1_Custom":
            self.criterion = L1CustomLoss(
                var_names=output_vars,
                train_weights=loss_config.get("train_weights", None),
            )
        elif loss_version == "MSE_Custom":
            self.criterion = MSECustomLoss(
                var_names=output_vars,
                train_weights=loss_config.get("train_weights", None),
            )
        elif loss_version == "Ramp":
            self.criterion = None
            self.wait_epochs = loss_config.get("wait_epochs", 10)
            self.ramp_epochs = loss_config.get("ramp_epochs", 10)
            self.ramp1 = nn.L1Loss()
            self.ramp2 = AreaWeightedClimateLoss(
                var_names=output_vars,
                train_weights=loss_config.get("train_weights", None),
            )
        else:
            raise NotImplementedError()
        self.loss_version = loss_version
        self.criterion1 = nn.L1Loss()
        self.criterion2 = AreaWeightedClimateLoss(
            var_names=output_vars
        )  # This is good to refine the std / time mean
        self.epochs_ran = 0
        self.normalizer = None
        self.val_preds, self.val_targets = [], []
        self.test_preds, self.test_targets = [], []

        self.train_preds, self.train_targets = [], []

    def forward(self, x):
        y = self.model(x)
        return y.squeeze(dim=1)

    def on_fit_start(self):
        self.normalizer = (
            self.trainer.datamodule.normalizer
        )  # Get the normalizer from the datamodule (see above)

    def training_step(self, batch, batch_idx):
        x, y = (
            batch  # Unpack inputs and targets (this is the output of the _getitem_ method in the Dataset above)
        )
        y_hat = self(x)  # Forward pass
        if self.criterion:
            loss = self.criterion(y_hat, y)
        else:
            if self.epochs_ran > self.wait_epochs + self.ramp_epochs:
                loss = self.ramp2(y_hat, y)
            elif self.epochs_ran > self.wait_epochs:
                loss = (
                    self.ramp1(y_hat, y)
                    * (1 - (self.epochs_ran - self.wait_epochs) / self.ramp_epochs)
                    + self.ramp2(y_hat, y)
                    * (self.epochs_ran - self.wait_epochs)
                    / self.ramp_epochs
                )
            else:
                loss = self.ramp1(y_hat, y)

        y_hat_np = self.normalizer.inverse_transform_output(
            y_hat.detach().cpu().numpy()
        )
        y_np = self.normalizer.inverse_transform_output(y.detach().cpu().numpy())

        self.train_preds.append(y_hat_np)
        self.train_targets.append(y_np)
        self.log("train/loss", loss, prog_bar=True)  # Log loss for tracking
        return loss

    def on_train_epoch_end(self):
        preds = np.concatenate(self.train_preds, axis=0)
        trues = np.concatenate(self.train_targets, axis=0)

        self._evaluate(preds, trues, phase="train")

        self.train_preds.clear()
        self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.criterion:
            loss = self.criterion(y_hat, y)
        else:
            if self.epochs_ran > self.wait_epochs + self.ramp_epochs:
                loss = self.ramp2(y_hat, y)
            elif self.epochs_ran > self.wait_epochs:
                loss = (
                    self.ramp1(y_hat, y)
                    * (1 - (self.epochs_ran - self.wait_epochs) / self.ramp_epochs)
                    + self.ramp2(y_hat, y)
                    * (self.epochs_ran - self.wait_epochs)
                    / self.ramp_epochs
                )
            else:
                loss = self.ramp1(y_hat, y)
        self.log("val/loss", loss * 10, prog_bar=True)

        y_hat_np = self.normalizer.inverse_transform_output(
            y_hat.detach().cpu().numpy()
        )
        y_np = self.normalizer.inverse_transform_output(y.detach().cpu().numpy())
        self.val_preds.append(y_hat_np)
        self.val_targets.append(y_np)

        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and ground truths from each val step/batch into one array
        preds = np.concatenate(self.val_preds, axis=0)
        trues = np.concatenate(self.val_targets, axis=0)
        self._evaluate(preds, trues, phase="val")
        np.save("val_preds.npy", preds)
        np.save("val_trues.npy", trues)
        self.val_preds.clear()
        self.val_targets.clear()
        self.epochs_ran += 1
        for i, g in enumerate(self.trainer.optimizers[0].param_groups):
            print(f"[Epoch {self.epochs_ran}] Current LR (group {i}): {g['lr']}")
            self.log_dict({"lr": g["lr"]})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_np = self.normalizer.inverse_transform_output(
            y_hat.detach().cpu().numpy()
        )
        y_np = y.detach().cpu().numpy()
        self.test_preds.append(y_hat_np)
        self.test_targets.append(y_np)

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths from each test step/batch into one array
        preds = np.concatenate(self.test_preds, axis=0)

        self._save_submission(preds)
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_config["lr"])
        scheduler_config = self.optimizer_config["scheduler"]
        if scheduler_config and scheduler_config["type"] == "ReduceLROnPlateau":
            print("using reduceLRonPlateau")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 1),
                min_lr=scheduler_config.get("min_lr", 1e-6),
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get("monitor", "val/loss"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if scheduler_config and scheduler_config["type"] == "OneCycle":
            print("Using one cycle LR")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config.get("max_lr"),
                epochs=scheduler_config.get("epochs"),
                steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                pct_start=scheduler_config.get("pct_start"),
                anneal_strategy=scheduler_config.get("anneal_strategy"),
                div_factor=scheduler_config.get("div_factor"),
                final_div_factor=scheduler_config.get("final_div_factor"),
                three_phase=scheduler_config.get("three_phase"),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    def _evaluate(self, preds, trues, phase="val"):
        datamodule = self.trainer.datamodule
        area_weights = datamodule.get_lat_weights()
        lat, lon = datamodule.get_coords()
        time = np.arange(preds.shape[0])
        output_vars = datamodule.output_vars

        device = "cuda"
        p_tensor = torch.tensor(preds, device=device)
        t_tensor = torch.tensor(trues, device=device)

        criterion1_score = self.criterion1(p_tensor, t_tensor).item()
        criterion2_score = self.criterion2(p_tensor, t_tensor, mode="val").item()
        print(
            f"[{phase.upper()}]: Criterion1 (MSE)={criterion1_score:.4f}, Criterion2 (Custom)={criterion2_score:.4f}"
        )
        self.log_dict(
            {
                f"{phase}/loss/crit1": criterion1_score,
                f"{phase}/loss/crit2": criterion2_score,
            }
        )

        for i, var in enumerate(output_vars):
            p = preds[:, i]
            t = trues[:, i]
            p_xr = xr.DataArray(
                p, dims=["time", "y", "x"], coords={"time": time, "y": lat, "x": lon}
            )
            t_xr = xr.DataArray(
                t, dims=["time", "y", "x"], coords={"time": time, "y": lat, "x": lon}
            )

            # RMSE
            rmse = np.sqrt(
                ((p_xr - t_xr) ** 2)
                .weighted(area_weights)
                .mean(("time", "y", "x"))
                .item()
            )
            # RMSE of time-mean
            mean_rmse = np.sqrt(
                ((p_xr.mean("time") - t_xr.mean("time")) ** 2)
                .weighted(area_weights)
                .mean(("y", "x"))
                .item()
            )
            # MAE of time-stddev
            std_mae = (
                np.abs(p_xr.std("time") - t_xr.std("time"))
                .weighted(area_weights)
                .mean(("y", "x"))
                .item()
            )

            print(
                f"[{phase.upper()}] {var}: RMSE={rmse:.4f}, Time-Mean RMSE={mean_rmse:.4f}, Time-Stddev MAE={std_mae:.4f}"
            )
            self.log_dict(
                {
                    f"{phase}/{var}/rmse": rmse,
                    f"{phase}/{var}/time_mean_rmse": mean_rmse,
                    f"{phase}/{var}/time_std_mae": std_mae,
                }
            )

    def _save_submission(self, predictions):
        datamodule = self.trainer.datamodule
        lat, lon = datamodule.get_coords()
        output_vars = datamodule.output_vars
        time = np.arange(predictions.shape[0])

        rows = []
        for t_idx, t in enumerate(time):
            for var_idx, var in enumerate(output_vars):
                for y_idx, y in enumerate(lat):
                    for x_idx, x in enumerate(lon):
                        row_id = f"t{t_idx:03d}_{var}_{y:.2f}_{x:.2f}"
                        pred = predictions[t_idx, var_idx, y_idx, x_idx]
                        rows.append({"ID": row_id, "Prediction": pred})

        df = pd.DataFrame(rows)
        os.makedirs("submissions", exist_ok=True)
        filepath = f"submissions/kaggle_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filepath, index=False)
        print(f"✅ Submission saved to: {filepath} with {len(rows)} rows")
