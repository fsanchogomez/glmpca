"""Tests for ``DaaremAccelerator``, on fixed points whose answer is known."""

from __future__ import annotations

import math

import pytest
import torch
from glmpca.acceleration import DaaremAccelerator

SIZE = 12


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)


def contraction() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A linear map `x -> A x + b` that converges slowly, and its fixed point.

    It seeds itself, so every caller in a test gets the same map.
    """
    torch.manual_seed(0)
    A = torch.randn(SIZE, SIZE) * 0.1
    A += torch.eye(SIZE) * 0.97
    b = torch.randn(SIZE)
    answer = torch.linalg.solve(torch.eye(SIZE) - A, b)
    return A, b, answer


def objective(A: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> float:
    """Rises as the iterate approaches the fixed point, so it can drive the monotonicity
    test the way a log-likelihood does."""
    return -float(((A @ x + b) - x).square().sum())


def run(*, accelerate: bool, steps: int = 60) -> tuple[torch.Tensor, DaaremAccelerator]:
    A, b, _ = contraction()
    accelerator = DaaremAccelerator(SIZE, order=5)
    x = torch.zeros(SIZE)
    current = objective(A, b, x)
    for _ in range(steps):
        stepped = A @ x + b
        plain = objective(A, b, stepped)
        if not accelerate:
            x, current = stepped, plain
            continue
        proposal = accelerator.propose(x, stepped - x)
        if proposal is None:
            x, current = stepped, plain
            continue
        candidate = objective(A, b, proposal)
        if candidate >= current - accelerator.mon_tol:
            x, current = proposal, candidate
            accelerator.accept(candidate)
        else:
            x, current = stepped, plain
            accelerator.reject(plain)
    return x, accelerator


def test_acceleration_reaches_the_fixed_point_sooner() -> None:
    _, _, answer = contraction()

    accelerated, accelerator = run(accelerate=True)
    plain, _ = run(accelerate=False)

    assert accelerator.accepted > 0
    # The plain iteration of this map is still far away after 60 steps.
    assert float((accelerated - answer).norm()) < float((plain - answer).norm()) / 100


def test_the_first_call_has_no_history_and_offers_nothing() -> None:
    accelerator = DaaremAccelerator(SIZE)

    assert accelerator.propose(torch.zeros(SIZE), torch.ones(SIZE)) is None
    assert accelerator.proposed == 0
    assert accelerator.propose(torch.ones(SIZE), torch.ones(SIZE)) is not None
    assert accelerator.proposed == 1


def test_a_rejected_jump_leaves_the_damping_alone() -> None:
    accelerator = DaaremAccelerator(SIZE)
    accelerator.propose(torch.zeros(SIZE), torch.ones(SIZE))
    accelerator.propose(torch.ones(SIZE), torch.ones(SIZE) * 0.5)

    accelerator.reject(1.0)

    assert accelerator.accepted == 0
    assert accelerator.shrink == 0


def test_an_accepted_jump_lowers_the_damping() -> None:
    accelerator = DaaremAccelerator(SIZE)
    accelerator.propose(torch.zeros(SIZE), torch.ones(SIZE))
    accelerator.propose(torch.ones(SIZE), torch.ones(SIZE) * 0.5)

    accelerator.accept(1.0)

    assert accelerator.accepted == 1
    assert accelerator.shrink == 1


def test_the_memory_wraps_at_the_order() -> None:
    order = 3
    accelerator = DaaremAccelerator(SIZE, order=order)
    theta = torch.zeros(SIZE)
    columns = []
    for step in range(order + 2):
        accelerator.propose(theta + step, torch.ones(SIZE) / (step + 1))
        columns.append(accelerator.column)
        accelerator.reject(-float(step))

    assert columns == [0, 1, 2, 0, 1]


def test_a_restart_that_lost_ground_damps_harder() -> None:
    order = 2
    accelerator = DaaremAccelerator(SIZE, order=order, kappa=25)
    theta = torch.zeros(SIZE)
    for step, value in enumerate((0.0, 5.0, 5.0, -100.0)):
        accelerator.propose(theta + step, torch.ones(SIZE) / (step + 1))
        accelerator.reject(value)

    # The objective at the second restart is below the one at the first, so the shrink
    # counter falls by the order of the accelerator.
    assert accelerator.shrink == -order


def test_the_damping_hits_its_target_share() -> None:
    accelerator = DaaremAccelerator(SIZE, order=4)
    values = torch.tensor([3.0, 1.0, 0.3, 0.05])
    projected = torch.tensor([2.0, 1.5, 0.7, 0.2]) ** 2

    ridge = accelerator._damping(projected, values)

    # The ridge should leave the damped coefficients at the share delta_k of the plain
    # least-squares ones, within the bracket that DampingFind stops on.
    target = math.exp(-0.5 * math.log1p(accelerator.alpha**accelerator.kappa))
    undamped = float((projected / values.square()).sum().sqrt())
    damped = float(
        (projected * (values / (values.square() + ridge)).square()).sum().sqrt()
    )
    low = math.exp(-0.5 * math.log1p(accelerator.alpha ** (accelerator.kappa + 0.5)))
    high = math.exp(-0.5 * math.log1p(accelerator.alpha ** (accelerator.kappa - 0.5)))
    assert low * undamped <= damped <= high * undamped
    assert abs(damped / undamped - target) < target


def test_an_order_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="order=0"):
        DaaremAccelerator(SIZE, order=0)


def test_the_order_never_exceeds_half_the_parameters() -> None:
    accelerator = DaaremAccelerator(4, order=10)

    assert accelerator.order == 2
    assert accelerator.iterates.shape == (4, 2)
