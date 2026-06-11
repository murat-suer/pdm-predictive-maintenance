"""Physics-Informed Neural Networks for RUL Prediction."""
import math

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """Combines MSE data loss with Weibull physics constraints."""

    def __init__(self, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def weibull_hazard(self, t: torch.Tensor, eta: float, beta: float) -> torch.Tensor:
        """Weibull hazard rate: h(t) = (beta/eta) * (t/eta)^(beta-1)"""
        return (beta / eta) * (t / eta) ** (beta - 1)

    def physics_loss(self, t: torch.Tensor, rul_pred: torch.Tensor,
                     eta: float, beta: float) -> torch.Tensor:
        """
        Physics constraints:
        1. RUL must decrease monotonically with age
        2. RUL must be non-negative
        3. RUL at t=0 should approximate Weibull mean life
        """
        # Monotonicity: d(RUL)/dt < 0
        # Use allow_unused=True to handle cases where t is not in the graph
        grad_outputs = torch.autograd.grad(
            rul_pred.sum(), t, create_graph=True, allow_unused=True
        )[0]
        if grad_outputs is not None:
            monotonicity_loss = torch.relu(grad_outputs).mean()  # Penalize positive gradients
        else:
            # If t is not connected to rul_pred, use finite differences
            # Sort by t and compute differences
            sorted_indices = torch.argsort(t)
            t_sorted = t[sorted_indices]
            rul_sorted = rul_pred[sorted_indices]
            if len(t_sorted) > 1:
                dt = t_sorted[1:] - t_sorted[:-1]
                drul = rul_sorted[1:] - rul_sorted[:-1]
                # Avoid division by zero
                safe_dt = torch.where(dt > 0, dt, torch.ones_like(dt))
                gradients = drul / safe_dt
                # Penalize positive gradients (RUL should decrease)
                monotonicity_loss = torch.relu(gradients).mean()
            else:
                monotonicity_loss = torch.tensor(0.0, device=rul_pred.device)

        # Non-negativity: RUL >= 0
        nonneg_loss = torch.relu(-rul_pred).mean()

        # Boundary condition: RUL(0) ≈ eta * gamma(1 + 1/beta)
        rul_at_zero = rul_pred[t == 0].mean() if (t == 0).any() else 0
        expected_mtl = eta * math.gamma(1 + 1/beta)
        boundary_loss = (rul_at_zero - expected_mtl) ** 2

        return monotonicity_loss + nonneg_loss + boundary_loss

    def forward(self, t: torch.Tensor, rul_pred: torch.Tensor,
                rul_true: torch.Tensor, eta: float, beta: float) -> torch.Tensor:
        data_loss = self.mse(rul_pred, rul_true)
        phys_loss = self.physics_loss(t, rul_pred, eta, beta)
        return data_loss + self.lambda_physics * phys_loss


class PINNRULPredictor(nn.Module):
    """Neural network with physics-informed constraints.

    Machine age ``t`` is a mandatory input alongside the sensor features:
    the monotonicity constraint d(RUL)/dt < 0 is only differentiable via
    autograd when t participates in the forward graph.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # +1 input for machine age t
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1)).squeeze(-1)
