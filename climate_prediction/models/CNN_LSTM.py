# THIS CODE IS ADAPTED FROM -> https://github.com/RolnickLab/ClimateSet

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """As converted from tf to torch, adapted from ClimateBench.
    Predicts single time step only #TODO we wanna change that do we? # tODO: documentation

    Original code below:

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2


    cnn_model = Sequential()
    cnn_model.add(Input(shape=(slider, 96, 144, 4)))
    cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
    cnn_model.add(TimeDistributed(AveragePooling2D(2)))
    cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
    cnn_model.add(LSTM(25, activation='relu'))
    cnn_model.add(Dense(1*96*144))
    cnn_model.add(Activation('linear'))
    cnn_model.add(Reshape((1, 96, 144)))"""

    def __init__(
        self,
        lon,
        lat,
        in_var_ids,
        out_var_ids,
        num_conv_filters: int = 192,
        lstm_hidden_size: int = 192,
        num_lstm_layers: int = 1,
        seq_to_seq: bool = False,
        seq_len: int = 10,
        dropout: float = 0.0,
        channels_last=True,
        datamodule_config=None,
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

        self.model = torch.nn.Sequential(
            # nn.Input(shape=(slider, width, height, num_input_vars)),
            TimeDistributed(
                nn.Conv2d(
                    in_channels=self.num_input_vars,
                    out_channels=num_conv_filters,
                    kernel_size=(3, 3),
                    padding="same",
                )
            ),  # we might need to permute because not channels last ?
            nn.ReLU(),  # , input_shape=(slider, width, height, num_input_vars)),
            TimeDistributed(nn.Dropout2d(dropout)),
            TimeDistributed(nn.AvgPool2d(2)),
            # TimeDistributed(nn.AdaptiveAvgPool1d(())), ##nGlobalAvgPool2d(), does not exist in pytorch
            TimeDistributed(nn.AvgPool2d((int(lat / 2), int(lon / 2)))),
            nn.Flatten(start_dim=2),
            nn.LSTM(
                input_size=num_conv_filters,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            ),  # returns tuple and complete sequence
            extract_tensor(seq_to_seq),  # ignore hidden and cell state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=lstm_hidden_size,
                out_features=self.num_output_vars * lon * lat,
            ),
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, X):
        x = X
        if self.channels_last:
            x = x.permute(
                (0, 1, 4, 2, 3)
            )  # torch con2d expects channels before height and witdth

        x = self.model(x)
        x = torch.reshape(
            x, (X.shape[0], self.out_seq_len, self.num_output_vars, self.lat, self.lon)
        )
        if self.channels_last:
            x = x.permute((0, 1, 3, 4, 2))

        return x


class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step"

    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*args)
        else:
            # only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(
                *[x.view(bs * seq_len, *x.shape[2:]) for x in args], **kwargs
            )
            out_shape = out.shape
            return out.view(bs, seq_len, *out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)

    def __repr__(self):
        return f"TimeDistributed({self.module})"


class extract_tensor(nn.Module):
    def __init__(self, seq_to_seq) -> None:
        super().__init__()
        self.seq_to_seq = seq_to_seq

    """ Helper Module to only extract output of a LSTM (ignore hidden and cell states)"""

    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        if not (self.seq_to_seq):
            tensor = tensor[:, -1, :]
        return tensor
