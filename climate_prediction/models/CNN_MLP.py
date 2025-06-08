import torch.nn as nn
import torch


class CNN_MLP(nn.Module):

    def __init__(
        self,
        lon,
        lat,
        in_var_ids,
        out_var_ids,
        hidden_size,
        seq_to_seq: bool = False,
        seq_len: int = 1,
        dropout: float = 0.0,
        channels_last=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_input_vars = len(in_var_ids)
        self.num_output_vars = len(out_var_ids)
        self.channels_last = channels_last

        self.lon = lon
        self.lat = lat
        self.channels_last = channels_last
        self.seq_len = seq_len

        if seq_to_seq:
            self.out_seq_len = self.seq_len
        else:
            self.out_seq_len = 1

        self.lin = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_vars,
                out_channels=(hidden_size),
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.AvgPool2d((lat, lon)),
            nn.Flatten(),
            nn.Linear(
                in_features=(hidden_size),
                out_features=hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=hidden_size, out_features=self.num_output_vars * lon * lat
            ),
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, X):
        B, T, F, H, W = X.shape
        x = X.reshape(B * T, F, H, W)
        x = self.lin(x)
        x = torch.reshape(
            x, (X.shape[0], self.out_seq_len, self.num_output_vars, self.lat, self.lon)
        )

        return x
