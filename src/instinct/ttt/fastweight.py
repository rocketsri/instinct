"""Branch-major nonlinear fast weights, with exact forking and rollback.

The oracle in :mod:`instinct.ttt.oracle` has to answer, for every candidate
write, "what would the memory look like if I applied this, and what would it
look like if I didn't?". Both branches must be evaluated, and then the memory
must be put back exactly as it was, because the next candidate is measured
against the *same* baseline. If rollback drifted even in the last bit, the
baseline would wander over the stream and the per-timestep utilities would stop
being comparable to each other — a failure that shows up as noise, not as a
crash.

So the state here is a plain container of tensors with a **leading branch
axis**, and three operations are kept deliberately dumb:

``snapshot`` / ``restore``
    ``clone`` out, ``copy_`` back. Same dtype, same shape, so restoration is
    bitwise, not approximately.
``fork``
    ``repeat_interleave`` on the branch axis. Forking is a copy, never a view:
    a view would let one branch's update alias into another's.
``forward``
    every matmul is a ``torch.bmm`` over the branch axis with *materialized*
    per-branch inputs.

That last point is what makes the load-bearing equivalence test pass. It is
tempting to broadcast a shared input across branches with ``einsum`` and skip
the copy; measured on this CPU, ``einsum`` with a broadcast operand does **not**
return bitwise-identical rows to the same contraction run one branch at a time,
while ``bmm`` on contiguous inputs does. Since the whole point of the oracle is
that a batched fork equals a sequential clone/apply/rollback, we pay for the
copy and keep the guarantee. (The copy is also cheap in the regime that matters:
``reports/t4_hardware.md`` measures this workload as launch-bound, with width
64 to 512 costing the same wall-clock.)

The state itself is a LaCT-style SwiGLU MLP,

    f(x) = W2 [ silu(W1 x) * (W3 x) ]

which is the smallest nonlinear memory that can actually be damaged by a write:
a linear associative memory's writes commute in a way that makes "destructive
plasticity" degenerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from instinct.core.rng import SeedScope

__all__ = [
    "FastWeightConfig",
    "FastWeightDelta",
    "FastWeightSnapshot",
    "FastWeights",
    "chunk_delta",
    "row_losses",
]


@dataclass(frozen=True, slots=True)
class FastWeightConfig:
    """Shape and step-size of the fast-weight memory.

    ``dtype`` defaults to float64 because phase zero is an *exactness*
    experiment run on CPU: the batched and sequential paths must agree, and
    float64 leaves no doubt about whether a disagreement is a bug or rounding.
    The GPU arm will set float16 (never bfloat16 — emulated and 9.5x slower on
    sm_75) and will have to re-establish equivalence at a stated tolerance.

    ``use_compile`` exists only so the flag has somewhere to live.
    ``torch.compile(mode="reduce-overhead")`` measured 0.45x on the T4 — slower,
    plus 17.4 s of compile time — so it stays off until it re-earns its place
    with a measurement on the larger arm.
    """

    d_in: int
    d_hidden: int
    d_out: int
    lr: float = 0.5
    # True by default, and that default is load-bearing. With unnormalized
    # updates the write_all reference trajectory diverges: measured on the
    # drifting-regression stream at lr=0.4 the primary-task loss reached 6.5e2
    # by step 8, 1e300 by step 11 and NaN by step 12. The oracle still returned
    # numbers -- plausible-looking, finite-until-they-weren't, and completely
    # meaningless. Normalizing the update keeps the reference roll bounded.
    normalize_update: bool = True
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    use_compile: bool = False

    def __post_init__(self) -> None:
        for name in ("d_in", "d_hidden", "d_out"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_hidden > 512:
            # Not a hard limit of the code, a limit of what the hardware note
            # licenses: at width <=512 capacity is free in wall-clock terms and
            # real work only starts showing at 1024.
            raise ValueError(
                f"d_hidden={self.d_hidden} exceeds the measured free-capacity width 512"
            )
        if self.use_compile:
            raise ValueError(
                "torch.compile measured 0.45x on the target GPU; "
                "enable only with a new profile"
            )


@dataclass(frozen=True, slots=True)
class FastWeightSnapshot:
    """Detached copies of every parameter, for exact restoration."""

    w1: Tensor
    w2: Tensor
    w3: Tensor


@dataclass(frozen=True, slots=True)
class FastWeightDelta:
    """A candidate write, shaped like the state it would be added to.

    Kept as its own type rather than a bare tuple so that the flattening used
    for norms, cosines and the retention-damage Gram matrix happens in exactly
    one place. Two deltas flattened in different orders would give a Gram
    matrix that is silently wrong rather than obviously wrong.
    """

    dw1: Tensor
    dw2: Tensor
    dw3: Tensor

    @property
    def n_branches(self) -> int:
        return int(self.dw1.shape[0])

    def flat(self) -> Tensor:
        """``(n_branches, n_params)``, in a fixed parameter order."""
        b = self.n_branches
        flat = [self.dw1.reshape(b, -1), self.dw2.reshape(b, -1), self.dw3.reshape(b, -1)]
        return torch.cat(flat, dim=1)

    def norm(self) -> Tensor:
        """Frobenius norm per branch, ``(n_branches,)``."""
        return torch.linalg.vector_norm(self.flat(), dim=1)

    def scale(self, factor: Tensor | float) -> FastWeightDelta:
        """Multiply every branch by a scalar or a per-branch factor."""
        if isinstance(factor, Tensor):
            f = factor.reshape(-1, 1, 1)
        else:
            f = torch.as_tensor(factor, dtype=self.dw1.dtype, device=self.dw1.device)
        return FastWeightDelta(self.dw1 * f, self.dw2 * f, self.dw3 * f)

    def select(self, index: Tensor) -> FastWeightDelta:
        return FastWeightDelta(
            self.dw1.index_select(0, index),
            self.dw2.index_select(0, index),
            self.dw3.index_select(0, index),
        )

    @staticmethod
    def cat(deltas: list[FastWeightDelta]) -> FastWeightDelta:
        return FastWeightDelta(
            torch.cat([d.dw1 for d in deltas], dim=0),
            torch.cat([d.dw2 for d in deltas], dim=0),
            torch.cat([d.dw3 for d in deltas], dim=0),
        )


class FastWeights:
    """A batch of SwiGLU fast-weight states sharing one leading branch axis.

    Branches are independent memories. Nothing in the forward pass mixes them,
    which is what lets one ``bmm`` evaluate a baseline and thirty-one
    counterfactual writes in a single pass.
    """

    __slots__ = ("cfg", "w1", "w2", "w3")

    def __init__(self, cfg: FastWeightConfig, w1: Tensor, w2: Tensor, w3: Tensor) -> None:
        if not (w1.shape[0] == w2.shape[0] == w3.shape[0]):
            raise ValueError("all parameters must share the leading branch axis")
        self.cfg = cfg
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # -- construction ------------------------------------------------------

    @staticmethod
    def init(cfg: FastWeightConfig, n_branches: int, scope: SeedScope) -> FastWeights:
        """Fresh memories, seeded from the repo's counter-based streams.

        Drawn through :mod:`instinct.core.rng` rather than a torch generator so
        that branch ``b``'s initialization depends on ``b`` alone: re-running
        with a different branch count leaves the surviving branches identical,
        which keeps paired comparisons across sweep configurations honest.
        """
        st = scope.stream("fastweight-init")
        lanes = np.arange(n_branches, dtype=np.int64)
        raw1 = st.normal(lanes, cfg.d_hidden * cfg.d_in, tick=1) / math.sqrt(cfg.d_in)
        raw2 = st.normal(lanes, cfg.d_out * cfg.d_hidden, tick=2) / math.sqrt(cfg.d_hidden)
        raw3 = st.normal(lanes, cfg.d_hidden * cfg.d_in, tick=3) / math.sqrt(cfg.d_in)

        def to_t(raw: np.ndarray, *shape: int) -> Tensor:
            t = torch.from_numpy(np.ascontiguousarray(raw)).to(dtype=cfg.dtype, device=cfg.device)
            return t.reshape(n_branches, *shape).contiguous()

        return FastWeights(
            cfg,
            to_t(raw1, cfg.d_hidden, cfg.d_in),
            to_t(raw2, cfg.d_out, cfg.d_hidden),
            to_t(raw3, cfg.d_hidden, cfg.d_in),
        )

    # -- branch algebra ----------------------------------------------------

    @property
    def n_branches(self) -> int:
        return int(self.w1.shape[0])

    def clone(self) -> FastWeights:
        return FastWeights(self.cfg, self.w1.clone(), self.w2.clone(), self.w3.clone())

    def snapshot(self) -> FastWeightSnapshot:
        return FastWeightSnapshot(
            self.w1.detach().clone(), self.w2.detach().clone(), self.w3.detach().clone()
        )

    def restore(self, snap: FastWeightSnapshot) -> None:
        """Put the state back exactly. ``copy_`` at equal dtype is bitwise."""
        self.w1.copy_(snap.w1)
        self.w2.copy_(snap.w2)
        self.w3.copy_(snap.w3)

    def fork(self, copies: int) -> FastWeights:
        """``copies`` independent duplicates of every branch, block-interleaved.

        Branch ``b`` of the original becomes branches ``b*copies .. b*copies+copies-1``.
        ``repeat_interleave`` allocates, deliberately: a stride-0 view would
        make one branch's in-place update visible to its siblings.
        """
        if copies < 1:
            raise ValueError(f"copies={copies} must be >= 1")
        return FastWeights(
            self.cfg,
            self.w1.repeat_interleave(copies, dim=0),
            self.w2.repeat_interleave(copies, dim=0),
            self.w3.repeat_interleave(copies, dim=0),
        )

    def select(self, index: Tensor) -> FastWeights:
        """Gather branches by index into a new, independently-owned state."""
        return FastWeights(
            self.cfg,
            self.w1.index_select(0, index).contiguous(),
            self.w2.index_select(0, index).contiguous(),
            self.w3.index_select(0, index).contiguous(),
        )

    @staticmethod
    def cat(states: list[FastWeights]) -> FastWeights:
        return FastWeights(
            states[0].cfg,
            torch.cat([s.w1 for s in states], dim=0),
            torch.cat([s.w2 for s in states], dim=0),
            torch.cat([s.w3 for s in states], dim=0),
        )

    # -- writes ------------------------------------------------------------

    def apply_(self, delta: FastWeightDelta, mask: Tensor | None = None) -> None:
        """Commit a write in place, optionally only on the masked branches.

        ``mask`` is a boolean ``(n_branches,)``. Masking rather than subsetting
        keeps every branch on the same code path, which is what makes the
        matched-compute claim in :mod:`instinct.ttt.frontier` true rather than
        aspirational: an admitted and a rejected write cost the same forward,
        the same backward, and the same add.
        """
        if delta.n_branches != self.n_branches:
            raise ValueError(f"delta has {delta.n_branches} branches, state has {self.n_branches}")
        if mask is None:
            self.w1.add_(delta.dw1)
            self.w2.add_(delta.dw2)
            self.w3.add_(delta.dw3)
            self._assert_finite()
            return
        m = mask.to(dtype=self.w1.dtype).reshape(-1, 1, 1)
        self.w1.add_(delta.dw1 * m)
        self.w2.add_(delta.dw2 * m)
        self.w3.add_(delta.dw3 * m)
        self._assert_finite()

    def _assert_finite(self) -> None:
        """Refuse to carry a diverged memory forward.

        A diverged fast-weight state does not announce itself. It keeps
        returning finite numbers for a while, then infinities, then NaNs, and an
        oracle scoring it produces a full table of plausible utilities that mean
        nothing at all. This turns that into an immediate, loud failure instead
        of a silent one, because the alternative is discovering it only when a
        downstream AUC comes out below chance and being unable to tell whether
        the hypothesis or the arithmetic was wrong.
        """
        for name, w in (("w1", self.w1), ("w2", self.w2), ("w3", self.w3)):
            if not bool(torch.isfinite(w).all()):
                raise FloatingPointError(
                    f"fast-weight {name} left the finite range after a write. "
                    "The reference trajectory has diverged, so every utility "
                    "computed from this point on is meaningless. Lower `lr` or "
                    "set `normalize_update=True`."
                )

    # -- forward -----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """``(B, N, d_in) -> (B, N, d_out)``, or ``(N, d_in)`` shared by all branches."""
        xb = broadcast_rows(x, self.n_branches)
        gate = torch.bmm(xb, self.w1.transpose(1, 2))
        up = torch.bmm(xb, self.w3.transpose(1, 2))
        hidden = torch.nn.functional.silu(gate) * up
        return torch.bmm(hidden, self.w2.transpose(1, 2))


def broadcast_rows(x: Tensor, n_branches: int) -> Tensor:
    """Materialize ``(N, d)`` as ``(n_branches, N, d)``, contiguous.

    ``expand`` would be free but produces stride-0 batches, and the batched
    path then stops being bitwise-comparable to the one-branch-at-a-time path.
    """
    if x.dim() == 3:
        if x.shape[0] != n_branches:
            raise ValueError(f"input has {x.shape[0]} branches, state has {n_branches}")
        return x.contiguous()
    if x.dim() != 2:
        raise ValueError(f"expected (N, d) or (B, N, d), got shape {tuple(x.shape)}")
    return x.unsqueeze(0).repeat(n_branches, 1, 1).contiguous()


def row_losses(state: FastWeights, x: Tensor, y: Tensor) -> Tensor:
    """Per-row squared error, ``(n_branches, N)``.

    Per row rather than per batch because the oracle needs the same forward to
    serve three different weightings — future chunks by horizon, aged probes by
    the retention term, and the cheap post-write audit — and reducing early
    would force three forwards where one suffices.
    """
    pred = state.forward(x)
    target = broadcast_rows(y, state.n_branches)
    return ((pred - target) ** 2).mean(dim=-1)


def chunk_delta(
    state: FastWeights,
    keys: Tensor,
    values: Tensor,
    *,
    lr: float | None = None,
    normalize: bool | None = None,
) -> FastWeightDelta:
    """The candidate write a LaCT chunk would produce: one gradient step.

    Branches are independent, so the per-branch losses are summed before
    ``backward`` and each branch still receives its own gradient. That is the
    difference between one backward pass and ``n_branches`` of them, and it is
    why the frontier replay can carry every (method, m) arm at once.

    ``normalize`` rescales each branch's write to a fixed Frobenius norm — a
    trust region. Off by default, because with it on the update norm carries no
    information and one of the cheap gate features the decision tree tests is
    identically constant.
    """
    cfg = state.cfg
    step = cfg.lr if lr is None else lr
    unit = cfg.normalize_update if normalize is None else normalize

    w1 = state.w1.detach().clone().requires_grad_(True)
    w2 = state.w2.detach().clone().requires_grad_(True)
    w3 = state.w3.detach().clone().requires_grad_(True)
    probe = FastWeights(cfg, w1, w2, w3)
    loss = row_losses(probe, keys, values).mean(dim=1).sum()
    g1, g2, g3 = torch.autograd.grad(loss, [w1, w2, w3])

    delta = FastWeightDelta(-g1, -g2, -g3)
    if unit:
        norm = delta.norm().clamp_min(torch.finfo(cfg.dtype).tiny)
        delta = delta.scale(1.0 / norm)
    return delta.scale(step)
