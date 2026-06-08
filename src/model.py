import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """LSTM cell where every matrix multiplication is replaced by a 2D convolution."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # The 4 gates (input, forget, output, cell) are produced by a single
        # convolution over the concatenation of x and h_prev, then split.
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, height, width, device=None):
        shape = (batch_size, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


class ConvLSTM(nn.Module):
    """Stack of ConvLSTMCell layers processed jointly across a sequence."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3, num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = len(hidden_channels)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                in_channels if i == 0 else hidden_channels[i - 1],
                hidden_channels[i],
                kernel_size,
            )
            for i in range(num_layers)
        ])

    def init_hidden_states(self, batch_size, height, width, device=None):
        return [cell.init_hidden(batch_size, height, width, device) for cell in self.cells]

    def step(self, x, hidden_states):
        """Run a single timestep of input x through the whole stack of cells."""
        new_hidden_states = []
        layer_input = x
        for cell, (h, c) in zip(self.cells, hidden_states):
            h, c = cell(layer_input, h, c)
            new_hidden_states.append((h, c))
            layer_input = h
        return new_hidden_states

    def forward(self, x, hidden_states=None):
        """
        Args:
            x: tensor of shape (batch, seq_len, channels, height, width)
            hidden_states: optional initial (h, c) pair per layer

        Returns:
            outputs: hidden states of the last layer at every timestep,
                shape (batch, seq_len, hidden_channels[-1], height, width)
            hidden_states: final (h, c) pair for each layer
        """
        batch_size, seq_len, _, height, width = x.shape

        if hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size, height, width, x.device)

        outputs = []
        for t in range(seq_len):
            hidden_states = self.step(x[:, t], hidden_states)
            outputs.append(hidden_states[-1][0])

        return torch.stack(outputs, dim=1), hidden_states


class PredictionModel(nn.Module):
    """Encodes input frames with a ConvLSTM, then predicts future frames autoregressively."""

    def __init__(self, in_channels=1, hidden_channels=(64, 64, 64), kernel_size=3):
        super().__init__()
        hidden_channels = list(hidden_channels)
        self.encoder = ConvLSTM(in_channels, hidden_channels, kernel_size)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, future_frames=10):
        """
        Args:
            x: input frames, shape (batch, seq_len, channels, height, width)
            future_frames: number of frames to predict autoregressively

        Returns:
            predicted frames, shape (batch, future_frames, 1, height, width)
        """
        _, hidden_states = self.encoder(x)

        frame = self.decoder_head(hidden_states[-1][0])
        predictions = [frame]

        for _ in range(future_frames - 1):
            hidden_states = self.encoder.step(frame, hidden_states)
            frame = self.decoder_head(hidden_states[-1][0])
            predictions.append(frame)

        return torch.stack(predictions, dim=1)
