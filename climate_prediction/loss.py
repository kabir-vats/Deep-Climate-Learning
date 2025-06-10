import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np


# CUSTOM LOSS FOR COMPETITION
class AreaWeightedClimateLoss(nn.Module):
    def __init__(self, var_names=["tas", "pr"], train_weights=None):
        """
        lat: 1D tensor of latitude values (length H)
        var_names: list of variable names corresponding to the output channels
        """
        super().__init__()
        print(var_names)
        self.lat = torch.tensor(np.linspace(-90, 90, 48), dtype=torch.float32)
        self.var_names = var_names
        self.var_weights = {"tas": 0.5, "pr": 0.5}
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = {
                "tas": {"monthly_rmse": 0.1, "time_mean": 1.0, "time_std": 1.0},
                "pr": {"monthly_rmse": 0.1, "time_mean": 1, "time_std": 0.75},
            }
        self.metric_weights = {
            "tas": {"monthly_rmse": 0.1, "time_mean": 1.0, "time_std": 1.0},
            "pr": {"monthly_rmse": 0.1, "time_mean": 1, "time_std": 0.75},
        }

        # Precompute normalized latitude weights
        lat_radians = torch.deg2rad(self.lat)
        weights = torch.cos(lat_radians)  # [H]
        self.register_buffer("lat_weights", weights / weights.sum())  # normalized

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mode="train"):
        """
        y_hat: [T, V, H, W]
        y:     [T, V, H, W]
        """
        T, V, H, W = y.shape
        total_score = 0.0

        for v_idx, var in enumerate(self.var_names):
            pred = y_hat[:, v_idx]  # [T, H, W]
            true = y[:, v_idx]  # [T, H, W]

            lat_w = self.lat_weights.view(1, H, 1)  # [1, H, 1]
            # Monthly RMSE (averaged over all)
            mse_t = F.mse_loss(pred, true, reduction="none")  # [T, H, W]
            mse_avg = torch.mean(
                torch.sum(mse_t * lat_w, dim=1)
            )  # sum over H, mean over T and W
            monthly_rmse = torch.sqrt(mse_avg)

            # Time-Mean RMSE (climatology bias)

            mean_pred = pred.mean(dim=0)  # [H, W]
            mean_true = true.mean(dim=0)

            time_mean_rmse = torch.sqrt(
                torch.mean(
                    torch.sum((mean_pred - mean_true) ** 2 * lat_w.squeeze(0), dim=0)
                )
            )  # mean over W

            # Time-Std MAE (variability)
            std_pred = pred.std(dim=0)  # [H, W]
            std_true = true.std(dim=0)
            time_std_mae = torch.mean(
                torch.sum(torch.abs(std_pred - std_true) * lat_w.squeeze(0), dim=0)
            )  # mean over W

            # Weighted score per variable
            if mode == "val":
                weights = self.metric_weights[var]
            else:
                weights = self.train_weights[var]
            var_score = (
                weights["monthly_rmse"] * monthly_rmse
                + weights["time_mean"] * time_mean_rmse
                + weights["time_std"] * time_std_mae
            )
            total_score += self.var_weights[var] * var_score

        return total_score


class L1CustomLoss(nn.Module):
    def __init__(self, var_names=["tas", "pr"], train_weights=None):
        super().__init__()
        self.lat = torch.tensor(np.linspace(-90, 90, 48), dtype=torch.float32)
        lat_radians = torch.deg2rad(self.lat)
        weights = torch.cos(lat_radians)  # [H]
        self.register_buffer("lat_weights", weights / weights.sum())
        self.var_names = var_names
        self.var_weights = {"tas": 0.5, "pr": 0.5}
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = {
                "tas": {"L1": 1, "time_mean": 1, "time_std": 1.0},
                "pr": {"L1": 1, "time_mean": 1, "time_std": 0.75},
            }

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mode="train"):
        """
        y_hat: [T, V, H, W]
        y:     [T, V, H, W]
        """
        T, V, H, W = y.shape
        total_score = 0.0

        for v_idx, var in enumerate(self.var_names):
            pred = y_hat[:, v_idx]  # [T, H, W]
            true = y[:, v_idx]  # [T, H, W]

            L1_loss = F.l1_loss(pred, true, reduction="mean")  # [T, H, W]

            std_pred = pred.std(dim=0)
            std_true = true.std(dim=0)
            time_std_mae = torch.mean(torch.abs(std_pred - std_true))  # mean over W

            mean_pred = pred.mean(dim=0)  # [H, W]
            mean_true = true.mean(dim=0)

            time_mean_rmse = torch.sqrt(torch.mean((mean_pred - mean_true) ** 2))

            var_score = (
                self.train_weights[var]["L1"] * L1_loss
                + self.train_weights[var]["time_mean"] * time_mean_rmse
                + self.train_weights[var]["time_std"] * time_std_mae
            )
            total_score += self.var_weights[var] * var_score
        return total_score


class MSECustomLoss(nn.Module):
    def __init__(self, var_names=["tas", "pr"], train_weights=None):
        super().__init__()
        self.lat = torch.tensor(np.linspace(-90, 90, 48), dtype=torch.float32)
        lat_radians = torch.deg2rad(self.lat)
        weights = torch.cos(lat_radians)  # [H]
        self.register_buffer("lat_weights", weights / weights.sum())
        self.var_names = var_names
        self.var_weights = {"tas": 0.5, "pr": 0.5}
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = {
                "tas": {"MSE": 0.1, "time_mean": 1, "time_std": 1.0},
                "pr": {"MSE": 0.1, "time_mean": 1, "time_std": 0.75},
            }

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mode="train"):
        """
        y_hat: [T, V, H, W]
        y:     [T, V, H, W]
        """
        T, V, H, W = y.shape
        total_score = 0.0

        for v_idx, var in enumerate(self.var_names):
            pred = y_hat[:, v_idx]  # [T, H, W]
            true = y[:, v_idx]  # [T, H, W]

            MSE_loss = F.mse_loss(pred, true, reduction="mean")  # [T, H, W]

            mean_pred = pred.mean(dim=0)  # [H, W]
            mean_true = true.mean(dim=0)

            time_mean_rmse = torch.sqrt(torch.mean((mean_pred - mean_true) ** 2))

            std_pred = pred.std(dim=0)
            std_true = true.std(dim=0)
            time_std_mae = torch.mean(torch.abs(std_pred - std_true))  # mean over W

            var_score = (
                self.train_weights[var]["MSE"] * MSE_loss
                + self.train_weights[var]["time_mean"] * time_mean_rmse
                + self.train_weights[var]["time_std"] * time_std_mae
            )
            total_score += self.var_weights[var] * var_score
        return total_score

class WeightedL1CustomLoss(nn.Module):
    def __init__(self, var_names=["tas", "pr"], train_weights=None):
        super().__init__()
        self.lat = torch.tensor(np.linspace(-90, 90, 48), dtype=torch.float32)
        lat_radians = torch.deg2rad(self.lat)
        weights = torch.cos(lat_radians)  # [H]
        self.register_buffer("lat_weights", weights / weights.sum())
        self.var_names = var_names
        self.var_weights = {"tas": 0.5, "pr": 0.5}
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = {
                "tas": {"L1": 1, "time_mean": 1, "time_std": 1.0},
                "pr": {"L1": 1, "time_mean": 1, "time_std": 0.75},
            }

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mode="train"):
        """
        y_hat: [T, V, H, W]
        y:     [T, V, H, W]
        """
        T, V, H, W = y.shape
        total_score = 0.0

        for v_idx, var in enumerate(self.var_names):
            pred = y_hat[:, v_idx]  # [T, H, W]
            true = y[:, v_idx]  # [T, H, W]

            lat_w = self.lat_weights.view(1, H, 1)  # [1, H, 1]
            # Monthly RMSE (averaged over all)
            

            L1_t = F.l1_loss(pred, true, reduction="none")  # [T, H, W]
            L1_loss = torch.mean(
                torch.sum(L1_t * lat_w, dim=1)
            )

            mean_pred = pred.mean(dim=0)  # [H, W]
            mean_true = true.mean(dim=0)

            time_mean_rmse = torch.sqrt(
                torch.mean(
                    torch.sum((mean_pred - mean_true) ** 2 * lat_w.squeeze(0), dim=0)
                )
            )  # mean over W

            # Time-Std MAE (variability)
            std_pred = pred.std(dim=0)  # [H, W]
            std_true = true.std(dim=0)
            time_std_mae = torch.mean(
                torch.sum(torch.abs(std_pred - std_true) * lat_w.squeeze(0), dim=0)
            )

            var_score = (
                self.train_weights[var]["L1"] * L1_loss
                + self.train_weights[var]["time_mean"] * time_mean_rmse
                + self.train_weights[var]["time_std"] * time_std_mae
            )
            total_score += self.var_weights[var] * var_score
        return total_score
