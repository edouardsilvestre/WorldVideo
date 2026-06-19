"""
ConvLSTM architecture adapted from OpenSTL (https://github.com/chengtan9907/OpenSTL),
licensed under Apache License 2.0. Reproduced here to load a pretrained checkpoint
(see results/best_model.pth provenance in report.md) since this project's own training
run could not be completed within available compute resources.

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach
for Precipitation Nowcasting", NeurIPS 2015.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width]),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width]),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
                nn.LayerNorm([num_hidden, height, width]),
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


def reshape_patch(x, patch_size):
    """(B, T, C, H, W) -> (B, T, patch_size^2 * C, H/patch_size, W/patch_size)."""
    b, t, c, h, w = x.shape
    p = patch_size
    x = x.view(b, t, c, h // p, p, w // p, p)
    x = x.permute(0, 1, 3, 5, 4, 6, 2)
    x = x.reshape(b, t, h // p, w // p, p * p * c)
    return x.permute(0, 1, 4, 2, 3)


def reshape_patch_back(x, patch_size):
    """(B, T, patch_size^2 * C, Hp, Wp) -> (B, T, C, Hp*patch_size, Wp*patch_size)."""
    b, t, ppc, hp, wp = x.shape
    p = patch_size
    c = ppc // (p * p)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.reshape(b, t, hp, wp, p, p, c)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(b, t, hp * p, wp * p, c)
    return x.permute(0, 1, 4, 2, 3)


class PredictionModel(nn.Module):
    """ConvLSTM next-frame predictor with patch embedding (OpenSTL ConvLSTM_Model)."""

    def __init__(self, in_channels=1, hidden_channels=(128, 128, 128, 128),
                 filter_size=5, stride=1, patch_size=4, layer_norm=False,
                 height=64, width=64):
        super().__init__()
        hidden_channels = list(hidden_channels)
        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.frame_channel = patch_size * patch_size * in_channels

        patch_height = height // patch_size
        patch_width = width // patch_size

        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                self.frame_channel if i == 0 else hidden_channels[i - 1],
                hidden_channels[i], patch_height, patch_width,
                filter_size, stride, layer_norm,
            )
            for i in range(self.num_layers)
        ])
        self.conv_last = nn.Conv2d(hidden_channels[-1], self.frame_channel,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, future_frames=10):
        """
        Args:
            x: input frames, shape (batch, seq_len, channels, height, width)
            future_frames: number of frames to predict autoregressively

        Returns:
            predicted frames, shape (batch, future_frames, channels, height, width)
        """
        batch, seq_len, _, height, width = x.shape
        device = x.device
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        x_patches = reshape_patch(x, self.patch_size)

        h_t = [torch.zeros(batch, hc, patch_height, patch_width, device=device)
               for hc in self.hidden_channels]
        c_t = [torch.zeros(batch, hc, patch_height, patch_width, device=device)
               for hc in self.hidden_channels]

        outputs = []
        x_gen = None
        for t in range(seq_len + future_frames - 1):
            net = x_patches[:, t] if t < seq_len else x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[-1])
            if t >= seq_len - 1:
                outputs.append(x_gen)

        out_patches = torch.stack(outputs, dim=1)
        return reshape_patch_back(out_patches, self.patch_size)
