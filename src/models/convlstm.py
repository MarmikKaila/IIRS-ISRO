"""ConvLSTM RNN for next-latent prediction with optional per-frame conditioning.

``LatentForecaster`` encodes an input sequence of SD-VAE latents (optionally
concatenated with extra per-frame conditioning channels, e.g. solar-zenith
angle for VIS) and autoregressively predicts future latents. Predictions are
*residual*: each output is the previous latent plus a learned delta, which
anchors the model to persistence — a strong nowcasting baseline.
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell: gates computed by one shared convolution.

    Uses reflect padding to avoid zero-padding edge artefacts that can show
    up as vertical/horizontal banding on satellite imagery.
    """

    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch,
                              kernel_size, padding=kernel_size // 2,
                              padding_mode="reflect")

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, batch, spatial, device):
        h = torch.zeros(batch, self.hidden_ch, *spatial, device=device)
        return h, h.clone()


class LatentForecaster(nn.Module):
    """Stacked ConvLSTM that forecasts future latents from a seed sequence.

    Input frame has ``latent_ch + cond_ch`` channels; the model predicts only
    the ``latent_ch`` channels (conditioning is supplied, not predicted).
    """

    def __init__(self, latent_ch=4, cond_ch=0, hidden_dims=(64, 64),
                 kernel_size=3, num_layers=None):
        super().__init__()
        hidden_dims = list(hidden_dims)
        if num_layers is not None:
            if num_layers < len(hidden_dims):
                hidden_dims = hidden_dims[:num_layers]
            while len(hidden_dims) < num_layers:
                hidden_dims.append(hidden_dims[-1])
        self.latent_ch = latent_ch
        self.cond_ch = cond_ch
        in_total = latent_ch + cond_ch
        self.layers = nn.ModuleList()
        for i, hid in enumerate(hidden_dims):
            in_ch = in_total if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(in_ch, hid, kernel_size))
        self.head = nn.Conv2d(hidden_dims[-1], latent_ch, kernel_size,
                              padding=kernel_size // 2,
                              padding_mode="reflect")

    def _step(self, inp, states):
        new_states, x = [], inp
        for layer, st in zip(self.layers, states):
            h, c = layer(x, st)
            new_states.append((h, c))
            x = h
        return new_states

    def forward(self, x, future_steps=1, future_cond=None,
                teacher=None, tf_ratio=0.0):
        """Forecast ``future_steps`` latents.

        x           : (B, T, latent_ch + cond_ch, H, W) seed (latents + cond).
        future_cond : (B, future_steps, cond_ch, H, W) future conditioning;
                      required when cond_ch > 0.
        teacher     : (B, S, latent_ch, H, W) optional ground-truth future
                      latents for scheduled sampling.
        returns     : (B, future_steps, latent_ch, H, W) predicted latents.
        """
        if self.cond_ch > 0 and future_cond is None:
            raise RuntimeError("future_cond is required when cond_ch > 0")

        B, T, C, H, W = x.shape
        states = [layer.init_state(B, (H, W), x.device) for layer in self.layers]
        for t in range(T):
            states = self._step(x[:, t], states)

        preds, base = [], x[:, -1, :self.latent_ch]      # last latent only
        for s in range(future_steps):
            delta = self.head(states[-1][0])
            pred = base + delta                          # (B, latent_ch, H, W)
            preds.append(pred)

            # pick the next latent (teacher or own prediction)
            if (teacher is not None and s < teacher.shape[1]
                    and float(torch.rand(())) < tf_ratio):
                nxt_lat = teacher[:, s]
            else:
                nxt_lat = pred

            # rebuild the next input (latent + conditioning if any)
            if self.cond_ch > 0:
                nxt = torch.cat([nxt_lat, future_cond[:, s]], dim=1)
            else:
                nxt = nxt_lat
            states = self._step(nxt, states)
            base = nxt_lat
        return torch.stack(preds, dim=1)


def build_model(cfg):
    """Instantiate a LatentForecaster from a channel config.

    cond_ch = 1 when ``day_night.enabled`` is true (solar-zenith-angle channel
    is added to the latent input); 0 otherwise.
    """
    tr = cfg["train"]
    cond_ch = 1 if cfg.get("day_night", {}).get("enabled", False) else 0
    return LatentForecaster(
        latent_ch=4,
        cond_ch=cond_ch,
        hidden_dims=tr["hidden_dims"],
        kernel_size=tr["kernel_size"],
        num_layers=tr["num_layers"],
    )
