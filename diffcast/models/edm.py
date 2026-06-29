"""EDM (Karras et al. 2022) — noise schedule, preconditioning, Heun sampler.

Implements the formulation from "Elucidating the Design Space of Diffusion-Based
Generative Models" (Karras, Aittala, Aila, Laine 2022, arXiv:2206.00364).

This file is purely numerical — it has no dependence on the model architecture
or the VAE. It's imported by ``diffusion_model.py`` which wraps a backbone
with this preconditioning.
"""
import torch


def edm_preconditioning(sigma, sigma_data=0.5):
    """Karras Table 1 preconditioning functions.

    sigma : (B,) tensor of noise levels.
    Returns four tensors broadcast-shaped to (B, 1, 1, 1, 1) for elementwise
    multiplication with (B, T, C, H, W) latents — except c_noise which is
    a flat (B,) embedding to feed the backbone's time conditioning.
    """
    sigma5 = sigma.view(-1, 1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma5 ** 2 + sigma_data ** 2)
    c_out = sigma5 * sigma_data / torch.sqrt(sigma_data ** 2 + sigma5 ** 2)
    c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma5 ** 2)
    c_noise = 0.25 * torch.log(sigma.flatten())          # (B,)
    return c_skip, c_out, c_in, c_noise


def sample_sigma_train(B, P_mean=-1.2, P_std=1.2, device="cuda", dtype=torch.float32):
    """Log-normal sampling of sigma at training time (paper Eq. 23)."""
    return (torch.randn(B, device=device, dtype=dtype) * P_std + P_mean).exp()


def edm_loss_weight(sigma, sigma_data=0.5):
    """Per-sample loss weighting (paper Eq. 8).
    Returns a (B, 1, 1, 1, 1) tensor."""
    sigma5 = sigma.view(-1, 1, 1, 1, 1)
    return (sigma5 ** 2 + sigma_data ** 2) / (sigma5 * sigma_data) ** 2


def karras_sigma_schedule(N, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                          device="cuda", dtype=torch.float32):
    """Karras step-size schedule for the Heun ODE sampler (paper Algorithm 1).

    Returns a (N+1,) tensor of sigma values from sigma_max down to 0.
    The trailing zero is the final step's target (clean sample).
    """
    ramp = torch.linspace(0, 1, N, device=device, dtype=dtype)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho
              + ramp * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
