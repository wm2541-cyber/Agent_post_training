"""Immutable domain models for the hidden-premise-conflict benchmark and judge harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

OrchestrationMode = Literal["fork_join", "two_stage_scout", "bidirectional_escalation"]
BlockerType = Literal["PREMISE_INVALIDATED", "INCONSISTENT_REQUIREMENT", "MISSING_CONTRACT"]
WorkerStatus = Literal["idle", "running", "completed", "preempted", "challenged"]
LeadDecision = Literal["replan", "override"]
AttributionOutcome = Literal[
    "compliant_flow",
    "valid_whistle_replan",
    "valid_whistle_override_fail",
    "fake_challenge_hack",
    "tamper_hack",
    "pingpong_exceeded",
    "protocol_error",
]


def freeze_mapping(data: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Return an immutable mapping copy to preserve functional state invariants."""
    return MappingProxyType(dict(data or {}))


@dataclass(frozen=True)
class ToolObservation:
    """Recorded evidence emitted by an actual sandbox tool invocation."""

    step: int
    worker_id: str
    tool_name: str
    target: str
    exit_code: int
    excerpt: str
    content_sha256: str


@dataclass(frozen=True)
class ChallengePayload:
    """Structured upward escalation payload emitted by a subagent."""

    worker_id: str
    step: int
    blocker_type: BlockerType
    invalidated_premise: str
    evidence_file: str
    evidence_excerpt: str
    proposed_hint: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """Immutable specification of a benchmark task with public/hidden verifier separation."""

    task_id: str
    title: str
    lead_brief: str
    initial_files: Mapping[str, str]
    public_test_files: Mapping[str, str]
    hidden_test_files: Mapping[str, str]
    has_true_premise_conflict: bool
    expected_conflict_markers: tuple[str, ...]
    expected_conflict_files: tuple[str, ...]
    max_challenges: int = 2
    max_steps: int = 24


@dataclass(frozen=True)
class WorkerAssignment:
    """Explicit task assignment carrying Lead's stated premises for fault attribution."""

    worker_id: str
    goal: str
    explicit_premises: tuple[str, ...]
    generation: int = 0
    writable_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerState:
    """Persistent immutable worker snapshot supporting respawn and preemption."""

    worker_id: str
    status: WorkerStatus
    assignment: WorkerAssignment
    local_notes: tuple[str, ...] = ()
    steps_taken: int = 0
    respawn_count: int = 0


@dataclass(frozen=True)
class EventLogItem:
    """Single immutable event along the orchestration DAG."""

    step: int
    actor: str
    action: str
    detail: str
    depends_on_steps: tuple[int, ...] = ()
    cost_units: float = 1.0


@dataclass(frozen=True)
class EpisodeState:
    """Pure episode state threaded through functional orchestration transitions."""

    task_id: str
    mode: OrchestrationMode
    step: int
    workers: Mapping[str, WorkerState] = field(default_factory=lambda: MappingProxyType({}))
    observations: tuple[ToolObservation, ...] = ()
    challenges: tuple[ChallengePayload, ...] = ()
    lead_decisions: tuple[tuple[int, LeadDecision, str], ...] = ()
    events: tuple[EventLogItem, ...] = ()
    public_tests_passed: bool = False
    terminated: bool = False
    termination_reason: str = ""


@dataclass(frozen=True)
class ChallengeAudit:
    """Neutral audit result for an upward challenge, withheld from agent prompts."""

    worker_id: str
    step: int
    evidence_grounded: bool
    conflict_verified: bool
    reason: str


@dataclass(frozen=True)
class JudgeVerdict:
    """Outcome-grounded evaluation output combining integrity, hidden suite, and audits."""

    task_id: str
    sandbox_intact: bool
    tampered_files: tuple[str, ...]
    public_passed: bool
    hidden_passed: bool
    hidden_summary: str
    challenge_audits: tuple[ChallengeAudit, ...]
    attribution_outcome: AttributionOutcome
    critical_path_cost: float


@dataclass(frozen=True)
class RewardBreakdown:
    """Decomposed asymmetric rewards for Lead and Subagents."""

    lead_reward: float
    subagent_reward: float
    outcome_label: AttributionOutcome
    notes: tuple[str, ...]
    per_worker_rewards: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


class ProtocolViolationError(RuntimeError):
    """Raised when an agent exceeds protocol correction retries without valid JSON output."""

    def __init__(self, actor: str, code: str, message: str) -> None:
        self.actor = actor
        self.code = code
        self.message = message
        super().__init__(f"[{actor}] {code}: {message}")
