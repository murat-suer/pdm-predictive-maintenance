"""Tests for Physics-Informed Neural Networks."""
import torch

from src.ml.pinn_rul import PhysicsInformedLoss, PINNRULPredictor


class TestPhysicsInformedLoss:
    def test_monotonicity_penalty(self):
        """RUL that increases with age should be penalized."""
        loss_fn = PhysicsInformedLoss(lambda_physics=1.0)
        t = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        rul_pred = torch.tensor([100.0, 150.0, 200.0], requires_grad=True)  # Wrong: increasing
        rul_true = torch.tensor([100.0, 80.0, 60.0])

        loss = loss_fn(t, rul_pred, rul_true, eta=500.0, beta=2.0)
        assert loss > 0, "Monotonicity violation should produce positive loss"

    def test_nonnegativity_penalty(self):
        """Negative RUL predictions should be penalized."""
        loss_fn = PhysicsInformedLoss(lambda_physics=1.0)
        t = torch.tensor([10.0], requires_grad=True)
        rul_pred = torch.tensor([-50.0], requires_grad=True)  # Wrong: negative
        rul_true = torch.tensor([100.0])

        loss = loss_fn(t, rul_pred, rul_true, eta=500.0, beta=2.0)
        assert loss > 0, "Negative RUL should produce positive loss"

    def test_physics_loss_decreases_with_correct_predictions(self):
        """Correct monotonic, non-negative RUL should have lower physics loss."""
        loss_fn = PhysicsInformedLoss(lambda_physics=1.0)
        t = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)

        # Correct: decreasing, non-negative
        rul_correct = torch.tensor([100.0, 80.0, 60.0], requires_grad=True)
        rul_true = torch.tensor([100.0, 80.0, 60.0])

        loss_correct = loss_fn(t, rul_correct, rul_true, eta=500.0, beta=2.0)

        # Wrong: increasing
        t2 = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        rul_wrong = torch.tensor([100.0, 150.0, 200.0], requires_grad=True)
        loss_wrong = loss_fn(t2, rul_wrong, rul_true, eta=500.0, beta=2.0)

        assert loss_correct < loss_wrong, "Correct predictions should have lower loss"


class TestPINNRULPredictor:
    def test_forward_pass(self):
        """Model should produce scalar RUL prediction."""
        model = PINNRULPredictor(input_dim=6)
        x = torch.randn(10, 6)  # Batch of 10, 6 features
        t = torch.rand(10) * 100
        rul = model(x, t)
        assert rul.shape == (10,), "Output should be batch of scalars"

    def test_training_loop(self):
        """Model should train without errors."""
        model = PINNRULPredictor(input_dim=6)
        loss_fn = PhysicsInformedLoss(lambda_physics=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Dummy data
        x = torch.randn(32, 6)
        t = torch.rand(32) * 100
        t.requires_grad = True
        rul_true = torch.rand(32) * 200

        # Training step
        optimizer.zero_grad()
        rul_pred = model(x, t)
        loss = loss_fn(t, rul_pred, rul_true, eta=500.0, beta=2.0)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_age_is_in_autograd_graph(self):
        """d(RUL)/dt must come from autograd, not the finite-difference fallback.

        Regression guard: if t is dropped from the model input again, this
        gradient comes back None and the physics loss silently degrades.
        """
        model = PINNRULPredictor(input_dim=6)
        x = torch.randn(16, 6)
        t = torch.rand(16) * 100
        t.requires_grad = True

        rul_pred = model(x, t)
        grad = torch.autograd.grad(rul_pred.sum(), t, allow_unused=True)[0]
        assert grad is not None, "t must be connected to the prediction graph"
        assert grad.shape == t.shape
