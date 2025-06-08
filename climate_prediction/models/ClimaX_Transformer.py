# THIS CODE IS FROM -> https://github.com/RolnickLab/ClimateSet

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from omegaconf import DictConfig


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_pytorch_stable(
    dim, timesteps, dtype=torch.float32, max_period=10000
):
    """
    Create sinusoidal timestep embeddings.
    Arguments:
        - `timesteps`: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        - `dim`: the dimension of the output.
        - `max_period`: controls the minimum frequency of the embeddings.
    Returns:
        - embedding: [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TokenizedBase(nn.Module):
    """Base model for tokenized MAE and tokenized ViT Including patch embedding and encoder."""

    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        init_mode="xavier",  # xavier or small
        in_vars=["pr", "tas"],
        channel_agg="mean",
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.init_mode = init_mode
        self.in_vars = in_vars

        # separate linear layers to embed each token, which is 1xpxp
        self.token_embeds = nn.ModuleList(
            [
                PatchEmbed(img_size, patch_size, 1, embed_dim)
                for i in range(len(in_vars))
            ]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # positional embedding and channel embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.channel_embed, self.channel_map = self.create_channel_embedding(
            learn_pos_emb, embed_dim
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        if channel_agg == "mean":
            self.channel_agg = None
        elif channel_agg == "attention":
            self.channel_agg = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )
            self.channel_query = nn.Parameter(
                torch.zeros(1, 1, embed_dim), requires_grad=True
            )
        else:
            raise NotImplementedError

        dpr = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

    def create_channel_embedding(self, learnable, dim):
        channel_embed = nn.Parameter(
            torch.zeros(1, len(self.in_vars), dim), requires_grad=learnable
        )
        # create a mapping from var --> idx
        channel_map = {}
        idx = 0
        for var in self.in_vars:
            channel_map[var] = idx
            idx += 1
        return channel_embed, channel_map

    def get_channel_ids(self, vars):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids)

    def get_channel_emb(self, channel_emb, vars):
        ids = self.get_channel_ids(vars)
        return channel_emb[:, ids, :]

    def initialize_weights(self):
        # initialization
        # initialize pos_emb and channel_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.in_vars))
        )
        self.channel_embed.data.copy_(
            torch.from_numpy(channel_embed).float().unsqueeze(0)
        )

        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            else:
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        pass

    def forward_loss(self, x):
        pass

    def forward(self, x):
        pass


""" Adapted from the ClimaX repository.
https://github.com/tung-nd/climax_all/blob/climatebench/src/models/components/tokenized_vit_continuous.py
"""


class TokenizedViTContinuous(TokenizedBase):
    def __init__(
        self,
        climate_modeling=True,  # always True with us...
        time_history=1,
        img_size=[128, 256],  # grid size
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        in_vars=["CO2", "SO2", "CH4", "BC"],
        out_vars=[
            "pr",
            "tas",
        ],  # is default vars for climate modelling #TODO: merge into a single variable
        channel_agg="mean",
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        init_mode="xavier",
        freeze_encoder: bool = False,
        nonlinear_head: bool = False,  # linear or nonlinear readout
    ):
        super().__init__(
            img_size,
            patch_size,
            drop_path,
            drop_rate,
            learn_pos_emb,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            init_mode,
            in_vars,
            channel_agg,
        )

        self.climate_modeling = climate_modeling
        self.freeze_encoder = freeze_encoder
        self.time_history = time_history
        self.img_size = img_size
        self.in_vars = in_vars
        self.out_vars = out_vars

        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, time_history, embed_dim), requires_grad=learn_pos_emb
        )

        # --------------------------------------------------------------------------
        # Decoder: either a linear or non linear prediction head
        # --------------------------------------------------------------------------
        for s in img_size:
            assert (s % patch_size) == 0, "Grid sizes must be  divisible by patch size."

        self.head = nn.ModuleList()
        if nonlinear_head:
            for i in range(decoder_depth):
                self.head.append(nn.Linear(embed_dim, embed_dim))
                self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.out_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        self.time_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.initialize_weights()

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                if "norm" in name:
                    continue
                else:
                    p.requires_grad_(False)

    def initialize_weights(self):
        # initialization
        super().initialize_weights()

        # time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.time_pos_embed.shape[-1], np.arange(self.time_history)
        )
        self.time_pos_embed.data.copy_(
            torch.from_numpy(time_pos_embed).float().unsqueeze(0)
        )

    def unpatchify(self, x, h=None, w=None):
        """
        x: (B, L, patch_size**2 *3)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        c = len(self.out_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_channel(self, x: torch.Tensor):
        """
        x: B, C, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # BxL, C, D

        if self.channel_agg is not None:
            channel_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
            x, _ = self.channel_agg(channel_query, x, x)  # BxL, D
            x = x.squeeze()
        else:
            x = torch.mean(x, dim=1)  # BxL, D

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def emb_lead_time(self, lead_times: torch.Tensor, embed_dim):
        # lead_times: B, 1
        sinusoidal_emb = get_1d_sincos_pos_embed_from_grid_pytorch_stable(
            embed_dim, lead_times, dtype=lead_times.dtype
        )
        return self.lead_time_embed(sinusoidal_emb)  # B, D

    def forward_encoder(self, x, lead_times, variables):
        """
        x: B, T, C, H, W
        """
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, C, H, W

        # embed tokens
        embeds = []
        var_ids = self.get_channel_ids(variables)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # BxT, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        channel_embed = self.get_channel_emb(self.channel_embed, variables)

        x = x + channel_embed.unsqueeze(2)  # BxT, C, L, D

        x = self.aggregate_channel(x)  # BxT, L, D

        x = x.unflatten(0, sizes=(b, t))  # B, T, L, D

        x = x + self.pos_embed.unsqueeze(1)
        # time emb: 1, T, D
        x = x + self.time_pos_embed.unsqueeze(2)

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D

        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
        x = x + lead_time_emb
        x = x.flatten(0, 1)  # BxT, L, D
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # BxT, L, D

        x = self.head(x)
        x = self.unpatchify(
            x
        )  # TODO not supported for grid sizes not divisible by patch size
        x = x.reshape(
            b, t, len(self.out_vars), self.img_size[0], self.img_size[1]
        )  # B T C H W

        return x

    def forward(self, x, lead_times):
        # x: N, T, C, H, W
        # y: N, 1/T, C, H, W

        preds = self.forward_encoder(x, lead_times, self.in_vars)

        return preds

    def predict(self, x, lead_times):
        with torch.no_grad():
            pred = self.forward_encoder(x, lead_times, self.in_vars)
        return pred


class ClimaX(nn.Module):
    def __init__(
        self,
        climate_modeling: bool = True,  # always True with us...
        time_history: int = 1,
        lon: int = 128,
        lat: int = 256,
        patch_size: int = 16,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        learn_pos_emb: bool = False,
        in_vars: List[str] = ["CO2", "SO2", "CH4", "BC"],
        out_vars: List[str] = ["pr", "tas"],  # is default vars for climate modelling
        channel_agg: str = "mean",
        embed_dim: int = 1024,
        depth: int = 24,
        decoder_depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_mode: str = "xavier",
        freeze_encoder: bool = False,
        datamodule_config: DictConfig = None,
        channels_last: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # get info from datamodule
        if datamodule_config is not None:
            if datamodule_config.get("out_var_ids") is not None:
                out_vars = datamodule_config.get("out_var_ids")
            if datamodule_config.get("in_var_ids") is not None:
                in_vars = datamodule_config.get("in_var_ids")
            if datamodule_config.get("channels_last") is not None:
                self.channels_last = datamodule_config.get("channels_last")
            if datamodule_config.get("lon") is not None:
                self.lon = datamodule_config.get("lon")
            if datamodule_config.get("lat") is not None:
                self.lat = datamodule_config.get("lat")

        else:
            self.lon = lon
            self.lat = lat
            self.channels_last = channels_last

        if climate_modeling:
            assert out_vars is not None
            self.out_vars = out_vars
        else:
            self.out_vars = in_vars

        img_size = [self.lat, self.lon]
        # create class
        self.model = TokenizedViTContinuous(
            climate_modeling=climate_modeling,
            time_history=time_history,
            img_size=img_size,  # grid size
            patch_size=patch_size,
            drop_path=drop_path,
            drop_rate=drop_rate,
            learn_pos_emb=learn_pos_emb,
            in_vars=in_vars,
            out_vars=out_vars,
            channel_agg=channel_agg,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_mode=init_mode,
            freeze_encoder=freeze_encoder,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        if self.channels_last:
            x = x.permute((0, 1, 4, 2, 3))

        lead_times = torch.zeros(x.shape[0]).to(
            x.device
        )  #  zero leadtimes for climate modelling task #TODO: remove?
        x = self.model.forward(x, lead_times)
        if self.channels_last:
            x = x.permute((0, 1, 3, 4, 2))
        x = x.nan_to_num()
        return x

    def get_patch_size(self):
        return self.model.patch_size
