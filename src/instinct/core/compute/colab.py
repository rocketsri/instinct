"""Dispatching GPU work to Colab, and the two things that actually kill it.

The only GPU available to this project is a Colab T4 reached through the `colab`
CLI, which is installed and authenticated. Everything here is built around three
measured facts about that CLI and one about unattended agents.

**1. Use ``colab run``, never ``colab new``.** ``colab run --gpu T4 script.py``
provisions a VM, executes the script, and tears the VM down — including when the
script raises. ``colab new`` leaves a session up, and an unstopped session keeps
its keep-alive daemon pinging until the 24-hour cap, burning compute units the
whole time. An agent that forgets to call ``colab stop`` therefore costs a day of
quota per mistake. ``colab run`` cannot make that mistake. This module builds no
``colab new`` command, and ``--keep`` defaults off.

**2. Do not poll for idle timeout.** The reflex is to poll the session and touch
it so Colab does not reap it for inactivity. That is already handled: ``colab
new`` (and ``run``) spawn a detached keep-alive daemon that does exactly this. A
watchdog that polls for idleness is duplicating a daemon that already works,
which is wasted effort at best and, at worst, a second process racing the first.

**3. Watch token health and daemon liveness instead.** Those are the failure
modes that actually end long unattended jobs, and neither looks like an idle
timeout:

  *Token expiry.* The OAuth access token lives about an hour (``colab whoami``
  reports the remaining time). The daemon refreshes it. When the refresh fails —
  a revoked grant, a missing ``colaboratory`` scope, a clock skew — the daemon
  starts getting 4xx responses, and after enough consecutive failures it logs
  ``keep_alive_stopped reason=consecutive_4xx_errors`` and exits. The VM then
  dies of idleness some minutes later, and the job is gone with it.

  *Daemon death.* The daemon can also stop for reasons unrelated to auth:
  ``session_not_found`` if state was written to a different ``--config`` path,
  ``endpoint_mismatch`` if the session was reassigned. Same outcome.

Both are visible before the VM dies: ``colab whoami`` reports the expiry,
``colab log -s NAME`` reports ``KEEP: error`` and ``KEEP: stopped reason=...``.
:class:`Watchdog` reads exactly those two signals and nothing else.

**4. Shard the job, and record the shards.** A preempted run must resume, not
restart, and a Colab VM is preemptible on a timescale shorter than a full sweep.
So work is split into shards with a persisted manifest, written atomically after
every state transition. A crash loses at most the shard in flight. Reconciliation
on resume is by config hash, not by index, so re-sharding a job does not silently
re-run finished work or skip new work.

Also worth knowing, because both cost a wasted provision:

* ``colab run --timeout`` defaults to **30 seconds** of execution. That is fine
  for a probe and lethal for a real shard, so :class:`ColabDispatcher` sets it
  explicitly and shards should be sized to fit inside it.
* An unrecognized ``--gpu`` value silently falls back to A100, which then fails
  at allocation. :data:`SUPPORTED_GPUS` is checked locally so a typo fails in
  milliseconds instead of after a provision round trip.

Nothing here launches a GPU job as a side effect of being imported or
constructed. The subprocess layer is a single injectable callable
(:data:`Runner`), which is what the tests replace.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from instinct.core.cache import key_for

__all__ = [
    "ColabDispatcher",
    "CommandResult",
    "DaemonHealth",
    "DispatchReport",
    "JobShard",
    "Runner",
    "ShardManifest",
    "TokenHealth",
    "Watchdog",
    "WatchdogVerdict",
    "parse_log",
    "parse_whoami",
    "subprocess_runner",
]

#: Accelerators the CLI recognizes. Anything else falls back to A100 silently.
SUPPORTED_GPUS = ("T4", "L4", "G4", "H100", "A100")
DEFAULT_GPU = "T4"

#: The CLI's own default execution timeout, kept here because it is a trap.
CLI_DEFAULT_TIMEOUT_S = 30.0
#: What we actually use for a shard. Shards are sized to fit; see the docstring.
DEFAULT_SHARD_TIMEOUT_S = 1800.0

#: Scopes the keep-alive daemon needs. A token missing `colaboratory` mints
#: fine and then 4xx-es on every keep-alive ping, which is the failure that
#: looks like an idle timeout and is not.
REQUIRED_SCOPES = ("https://www.googleapis.com/auth/colaboratory",)

#: Refuse to start a shard with less than this much token life left. A shard
#: that outlives its token depends on a refresh that may not happen.
DEFAULT_MIN_TOKEN_REMAINING_S = 900.0


# ---------------------------------------------------------------------------
# The subprocess seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


#: Everything that talks to the outside world goes through this one callable, so
#: a test can substitute a table of canned outputs and no test can accidentally
#: provision a VM.
Runner = Callable[[Sequence[str], float | None], CommandResult]


def subprocess_runner(argv: Sequence[str], timeout: float | None = None) -> CommandResult:
    """Default :data:`Runner`: run the command, capture everything, never raise.

    A non-zero exit is data, not an exception: the dispatcher records it against
    the shard and moves on, and a shard that failed once is worth retrying. A
    timeout is reported as return code 124, matching ``timeout(1)``, so the two
    kinds of failure are distinguishable in the manifest.
    """
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=tuple(argv),
            returncode=124,
            stdout=_decode(exc.stdout),
            stderr=f"timed out after {timeout}s",
            duration_s=time.perf_counter() - start,
        )
    except OSError as exc:
        return CommandResult(
            argv=tuple(argv),
            returncode=127,
            stdout="",
            stderr=str(exc),
            duration_s=time.perf_counter() - start,
        )
    return CommandResult(
        argv=tuple(argv),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=time.perf_counter() - start,
    )


def _decode(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Signal 1: token health
# ---------------------------------------------------------------------------

_EXPIRES_RE = re.compile(r"^\s*Expires in:\s*(.+?)\s*$", re.MULTILINE)
_EMAIL_RE = re.compile(r"^\s*Email:\s*(.+?)\s*$", re.MULTILINE)
_DURATION_RE = re.compile(r"(\d+)\s*([dhms])")
_UNIT_SECONDS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> float:
    """Parse ``colab whoami``'s human durations: ``59m``, ``1h 2m``, ``expired``.

    Returns seconds; ``0.0`` for anything that reads as expired, and ``-1.0``
    when the text is unparseable. An unparseable expiry is not treated as
    healthy — an unknown token lifetime is a reason to check, not to proceed.
    """
    lowered = text.strip().lower()
    if not lowered or "expired" in lowered:
        return 0.0
    parts = _DURATION_RE.findall(lowered)
    if not parts:
        return -1.0
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in parts)


@dataclass(frozen=True, slots=True)
class TokenHealth:
    email: str
    expires_in_s: float
    scopes: tuple[str, ...]
    raw: str = ""

    def missing_scopes(self, required: Sequence[str] = REQUIRED_SCOPES) -> tuple[str, ...]:
        return tuple(s for s in required if s not in self.scopes)

    def healthy(
        self,
        *,
        min_remaining_s: float = DEFAULT_MIN_TOKEN_REMAINING_S,
        required: Sequence[str] = REQUIRED_SCOPES,
    ) -> bool:
        return self.expires_in_s >= min_remaining_s and not self.missing_scopes(required)


def parse_whoami(text: str) -> TokenHealth:
    """Read ``colab whoami`` output into a :class:`TokenHealth`.

    Parsing human-readable CLI output is fragile, and it is what is available —
    there is no machine-readable form of this command. The fragility is bounded
    by treating an unparsed expiry as ``-1`` (unhealthy) rather than as a large
    number, so a CLI format change fails closed.
    """
    email_m = _EMAIL_RE.search(text)
    expires_m = _EXPIRES_RE.search(text)
    scopes: list[str] = []
    in_scopes = False
    for line in text.splitlines():
        if line.strip().startswith("Scopes:"):
            in_scopes = True
            continue
        if in_scopes:
            stripped = line.strip()
            if stripped.startswith("- "):
                scopes.append(stripped[2:].strip())
            elif stripped:
                in_scopes = False
    return TokenHealth(
        email=email_m.group(1) if email_m else "",
        expires_in_s=parse_duration(expires_m.group(1)) if expires_m else -1.0,
        scopes=tuple(scopes),
        raw=text,
    )


# ---------------------------------------------------------------------------
# Signal 2: keep-alive daemon liveness
# ---------------------------------------------------------------------------

_KEEP_STARTED_RE = re.compile(r"KEEP:\s*started")
_KEEP_ERROR_RE = re.compile(r"KEEP:\s*error\b.*?status=(\S+)")
_KEEP_STOPPED_RE = re.compile(r"KEEP:\s*stopped\b.*?reason=(\S+)")
_TERMINATED_RE = re.compile(r"EVENT:\s*session_terminated")


@dataclass(frozen=True, slots=True)
class DaemonHealth:
    """What ``colab log -s NAME`` says about the keep-alive daemon."""

    session: str
    started: bool
    stopped: bool
    stop_reason: str
    terminated: bool
    error_count: int
    last_status: str
    last_line: str

    @property
    def alive(self) -> bool:
        """Started, not stopped, and the session was not torn down.

        A terminated session is not a failure — that is what ``colab run`` does
        on success — so callers distinguish ``alive`` from ``terminated`` rather
        than treating both as trouble.
        """
        return self.started and not self.stopped and not self.terminated


def parse_log(session: str, text: str) -> DaemonHealth:
    """Parse the ``colab log`` event stream for keep-alive state.

    Only the *last* stop matters, and only stops after the last start: a
    long-running session can legitimately have an old stopped daemon in its
    history if it was restarted.
    """
    started = stopped = terminated = False
    reason = ""
    errors = 0
    last_status = ""
    last_line = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        last_line = line.strip()
        if _KEEP_STARTED_RE.search(line):
            started, stopped, reason = True, False, ""
            continue
        err = _KEEP_ERROR_RE.search(line)
        if err:
            errors += 1
            last_status = err.group(1)
            continue
        stop = _KEEP_STOPPED_RE.search(line)
        if stop:
            stopped, reason = True, stop.group(1)
            continue
        if _TERMINATED_RE.search(line):
            terminated = True
    return DaemonHealth(
        session=session,
        started=started,
        stopped=stopped,
        stop_reason=reason,
        terminated=terminated,
        error_count=errors,
        last_status=last_status,
        last_line=last_line,
    )


@dataclass(frozen=True, slots=True)
class WatchdogVerdict:
    ok: bool
    reasons: tuple[str, ...]
    token: TokenHealth | None = None
    daemon: DaemonHealth | None = None

    def raise_if_unhealthy(self) -> None:
        if not self.ok:
            raise RuntimeError("colab watchdog: " + "; ".join(self.reasons))


@dataclass
class Watchdog:
    """Checks the two things that end long unattended Colab jobs.

    Deliberately *not* a poller. It has no loop and no sleep, because the only
    thing a loop would buy is the idle-timeout behaviour the CLI's own keep-alive
    daemon already provides. The dispatcher calls :meth:`check` between shards,
    which is when the answer can still change an outcome: a shard not started is
    a shard that resumes cleanly later.
    """

    runner: Runner = subprocess_runner
    min_token_remaining_s: float = DEFAULT_MIN_TOKEN_REMAINING_S
    required_scopes: tuple[str, ...] = REQUIRED_SCOPES
    command_timeout_s: float = 60.0

    def token(self) -> TokenHealth:
        result = self.runner(["colab", "whoami"], self.command_timeout_s)
        if not result.ok:
            return TokenHealth(email="", expires_in_s=-1.0, scopes=(), raw=result.stderr)
        return parse_whoami(result.stdout)

    def daemon(self, session: str) -> DaemonHealth:
        result = self.runner(["colab", "log", "-s", session], self.command_timeout_s)
        if not result.ok:
            return DaemonHealth(
                session=session,
                started=False,
                stopped=False,
                stop_reason="log_unavailable",
                terminated=False,
                error_count=0,
                last_status="",
                last_line=result.stderr.strip(),
            )
        return parse_log(session, result.stdout)

    def check(self, session: str | None = None) -> WatchdogVerdict:
        """Token health always; daemon liveness only when a session is named.

        ``colab run`` sessions are ephemeral and usually gone by the time anyone
        asks, so the daemon check is opt-in. The token check is not: a token
        about to expire will take down the *next* shard too.
        """
        reasons: list[str] = []
        tok = self.token()
        if tok.expires_in_s < 0:
            reasons.append(
                "cannot read token expiry from `colab whoami` (auth broken, or the CLI "
                "changed its output format); treating as unhealthy"
            )
        elif tok.expires_in_s < self.min_token_remaining_s:
            reasons.append(
                f"access token expires in {tok.expires_in_s:.0f}s, below the "
                f"{self.min_token_remaining_s:.0f}s floor: if the refresh fails the "
                "keep-alive daemon will 4xx and die, taking the VM with it"
            )
        missing = tok.missing_scopes(self.required_scopes)
        if missing:
            reasons.append(
                f"token is missing scope(s) {list(missing)}: keep-alive pings will 4xx. "
                "Re-authenticate with scripts/colab_auth_start.sh"
            )

        dae: DaemonHealth | None = None
        if session is not None:
            dae = self.daemon(session)
            if dae.stopped:
                reasons.append(
                    f"keep-alive daemon for {session!r} stopped: reason={dae.stop_reason}"
                    + (f", last status={dae.last_status}" if dae.last_status else "")
                )
            elif dae.error_count:
                reasons.append(
                    f"keep-alive daemon for {session!r} logged {dae.error_count} error(s), "
                    f"last status={dae.last_status or 'unknown'}: refresh is failing"
                )
        return WatchdogVerdict(ok=not reasons, reasons=tuple(reasons), token=tok, daemon=dae)


# ---------------------------------------------------------------------------
# Sharded job manifests
# ---------------------------------------------------------------------------

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass
class JobShard:
    """One unit of dispatchable work, and its state across restarts."""

    index: int
    config_hash: str
    config: Any = None
    status: str = PENDING
    attempts: int = 0
    session: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    returncode: int | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "config_hash": self.config_hash,
            "config": self.config,
            "status": self.status,
            "attempts": self.attempts,
            "session": self.session,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "returncode": self.returncode,
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> JobShard:
        return JobShard(
            index=int(data["index"]),
            config_hash=str(data["config_hash"]),
            config=data.get("config"),
            status=str(data.get("status", PENDING)),
            attempts=int(data.get("attempts", 0)),
            session=str(data.get("session", "")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            returncode=data.get("returncode"),
            error=str(data.get("error", "")),
        )


@dataclass
class ShardManifest:
    """Persisted state of a sharded job, so a preemption resumes.

    Written atomically after every transition. The cost is one small file write
    per shard, which is nothing next to a GPU shard, and the benefit is that a
    process killed at any instant leaves a manifest that is either the state
    before the transition or the state after — never a truncated file that the
    resume path cannot read.
    """

    path: Path
    job_id: str
    shards: list[JobShard] = field(default_factory=list)
    created_at: str = ""

    # -- construction -----------------------------------------------------

    @staticmethod
    def plan(
        path: str | Path,
        job_id: str,
        configs: Sequence[Mapping[str, Any]],
        *,
        code_version: str = "",
    ) -> ShardManifest:
        """Create the manifest, or reconcile an existing one against ``configs``.

        Reconciliation is by config hash, never by index. Re-sharding a job (a
        different chunk size, an added grid point) reorders indices, and an
        index-keyed resume would then mark the wrong work as finished — the worst
        possible failure for a resume path, because it produces a complete-looking
        run with missing cells.

        Shards left in ``running`` by a killed process are reset to ``pending``
        with their attempt count intact. The alternative — trusting ``running``
        — hangs the resume forever on work nobody is doing.
        """
        p = Path(path)
        hashes = [key_for(cfg, code_version=code_version) for cfg in configs]
        previous: dict[str, JobShard] = {}
        created = datetime.now(UTC).isoformat(timespec="seconds")
        if p.exists():
            old = ShardManifest.load(p)
            created = old.created_at or created
            previous = {s.config_hash: s for s in old.shards}

        shards: list[JobShard] = []
        for i, (cfg, h) in enumerate(zip(configs, hashes)):
            prior = previous.get(h)
            if prior is None:
                shards.append(JobShard(index=i, config_hash=h, config=dict(cfg)))
                continue
            status = PENDING if prior.status == RUNNING else prior.status
            shards.append(
                JobShard(
                    index=i,
                    config_hash=h,
                    config=dict(cfg),
                    status=status,
                    attempts=prior.attempts,
                    session=prior.session,
                    started_at=prior.started_at,
                    finished_at=prior.finished_at,
                    duration_s=prior.duration_s,
                    returncode=prior.returncode,
                    error=prior.error,
                )
            )
        manifest = ShardManifest(path=p, job_id=job_id, shards=shards, created_at=created)
        manifest.save()
        return manifest

    @staticmethod
    def load(path: str | Path) -> ShardManifest:
        p = Path(path)
        data = json.loads(p.read_text())
        return ShardManifest(
            path=p,
            job_id=str(data.get("job_id", "")),
            shards=[JobShard.from_dict(s) for s in data.get("shards", [])],
            created_at=str(data.get("created_at", "")),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "job_id": self.job_id,
                "created_at": self.created_at,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "shards": [s.as_dict() for s in self.shards],
            },
            indent=2,
            sort_keys=True,
        )
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, self.path)

    # -- state transitions ------------------------------------------------

    def pending(self, *, include_failed: bool = True) -> list[JobShard]:
        """Shards still owing work, in index order.

        Failed shards are included by default: a Colab failure is far more often
        a preemption or a provisioning hiccup than a bug in the shard, and a
        resume that skips them silently drops cells from the grid. Pass
        ``include_failed=False`` when a failure means "do not try again".
        """
        wanted = {PENDING, FAILED} if include_failed else {PENDING}
        return [s for s in self.shards if s.status in wanted]

    def start(self, shard: JobShard, *, session: str = "") -> None:
        shard.status = RUNNING
        shard.attempts += 1
        shard.session = session
        shard.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        shard.error = ""
        self.save()

    def finish(self, shard: JobShard, result: CommandResult) -> None:
        shard.status = DONE if result.ok else FAILED
        shard.returncode = result.returncode
        shard.duration_s = result.duration_s
        shard.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        shard.error = "" if result.ok else (result.stderr or result.stdout)[-2000:]
        self.save()

    def progress(self) -> dict[str, int]:
        counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
        for s in self.shards:
            counts[s.status] = counts.get(s.status, 0) + 1
        counts["total"] = len(self.shards)
        return counts


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchReport:
    job_id: str
    attempted: int
    completed: int
    failed: int
    skipped: int
    aborted_reason: str = ""
    commands: tuple[str, ...] = ()

    def __str__(self) -> str:
        tail = f", aborted: {self.aborted_reason}" if self.aborted_reason else ""
        return (
            f"{self.job_id}: attempted {self.attempted}, completed {self.completed}, "
            f"failed {self.failed}, skipped {self.skipped}{tail}"
        )


@dataclass
class ColabDispatcher:
    """Builds and runs one ``colab run`` per shard.

    One VM per shard, provisioned and destroyed by the CLI. That is more
    provisioning overhead than reusing a session, and it is the right trade: a
    reused session is a session somebody has to remember to stop, and the cost of
    forgetting is a day of compute units.
    """

    script: Path
    gpu: str = DEFAULT_GPU
    runner: Runner = subprocess_runner
    watchdog: Watchdog | None = None
    timeout_s: float = DEFAULT_SHARD_TIMEOUT_S
    env: Mapping[str, str] = field(default_factory=dict)
    keep: bool = False  # see the module docstring; leaving sessions up costs quota

    def __post_init__(self) -> None:
        if self.gpu not in SUPPORTED_GPUS:
            raise ValueError(
                f"unsupported --gpu {self.gpu!r}. The CLI silently falls back to A100 for "
                f"unrecognized values and then fails at allocation, so this is checked "
                f"locally. Supported: {list(SUPPORTED_GPUS)}"
            )
        self.script = Path(self.script)

    def argv_for(self, shard: JobShard, *, session: str | None = None) -> list[str]:
        """The exact command line for one shard.

        The shard's config travels as a single JSON argument rather than as a
        pile of flags. The remote script parses one thing, and the argument is
        byte-identical to what the manifest recorded, so "what was this shard
        actually run with" has one answer.
        """
        argv = ["colab", "run", "--gpu", self.gpu, "--timeout", str(self.timeout_s)]
        if session:
            argv += ["--session", session]
        if self.keep:
            argv += ["--keep"]
        for key, value in sorted(self.env.items()):
            argv += ["--env", f"{key}={value}"]
        argv += [str(self.script), "--shard", str(shard.index), "--config-json"]
        argv += [json.dumps(shard.config, sort_keys=True, separators=(",", ":"))]
        return argv

    def dispatch(
        self,
        manifest: ShardManifest,
        *,
        max_shards: int | None = None,
        dry_run: bool = False,
        include_failed: bool = True,
    ) -> DispatchReport:
        """Run pending shards, checking the watchdog before each one.

        The watchdog is checked *before* a shard rather than during it, because
        that is the only point where the answer is actionable: a shard not
        started stays pending and resumes later, whereas a shard aborted halfway
        has burned a provision for nothing. When the check fails the loop stops
        rather than continuing to the next shard — a dead token or a dead daemon
        will not fix itself, and grinding through the remaining shards would
        turn one failure into a whole failed sweep.
        """
        todo = manifest.pending(include_failed=include_failed)
        if max_shards is not None:
            todo = todo[:max_shards]

        attempted = completed = failed = 0
        commands: list[str] = []
        aborted = ""

        for shard in todo:
            if self.watchdog is not None:
                verdict = self.watchdog.check()
                if not verdict.ok:
                    aborted = "; ".join(verdict.reasons)
                    break

            argv = self.argv_for(shard, session=f"{manifest.job_id}-{shard.index:04d}")
            commands.append(shlex.join(argv))
            if dry_run:
                continue

            attempted += 1
            manifest.start(shard, session=f"{manifest.job_id}-{shard.index:04d}")
            result = self.runner(argv, self.timeout_s + 300.0)
            manifest.finish(shard, result)
            if result.ok:
                completed += 1
            else:
                failed += 1

        return DispatchReport(
            job_id=manifest.job_id,
            attempted=attempted,
            completed=completed,
            failed=failed,
            skipped=len(todo) - attempted,
            aborted_reason=aborted,
            commands=tuple(commands),
        )
