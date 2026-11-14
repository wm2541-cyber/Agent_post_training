"""Turn-based Interactive Arena CLI & persistent JSON state machine for Human vs LLM matches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from hidden_premise_lab.judge import evaluate_episode
from hidden_premise_lab.live_llm import (
    LiveLLMPolicyTeam,
    append_jsonl_log,
    build_lead_init_prompt,
    build_lead_response_prompt,
    build_worker_action_prompt,
    collect_scout_observations,
    parse_assignments,
    parse_lead_response,
    parse_worker_action,
)
from hidden_premise_lab.models import (
    AttributionOutcome,
    BlockerType,
    ChallengeAudit,
    ChallengePayload,
    EpisodeState,
    EventLogItem,
    JudgeVerdict,
    LeadDecision,
    OrchestrationMode,
    ProtocolViolationError,
    RewardBreakdown,
    TaskSpec,
    ToolObservation,
    WorkerAssignment,
    WorkerState,
    WorkerStatus,
    freeze_mapping,
)
from hidden_premise_lab.orchestrator import (
    LeadResponse,
    ToolAction,
    WorkerAction,
    apply_initial_assignments,
    apply_lead_response,
    create_scout_state,
    execute_worker_turn_step,
    freeze_mapping_generic,
    terminate_protocol_error,
)
from hidden_premise_lab.reward import compute_rewards
from hidden_premise_lab.sandbox import SandboxSession
from hidden_premise_lab.tasks import all_benchmark_tasks, get_task_by_id

ARENA_SCHEMA_VERSION = 1

ArenaPhase = Literal[
    "awaiting_lead_init",
    "awaiting_worker",
    "awaiting_lead_response",
    "finished",
]

VALID_PHASES: tuple[ArenaPhase, ...] = (
    "awaiting_lead_init",
    "awaiting_worker",
    "awaiting_lead_response",
    "finished",
)
VALID_MODES: tuple[OrchestrationMode, ...] = (
    "fork_join",
    "two_stage_scout",
    "bidirectional_escalation",
)


VALID_BLOCKERS: tuple[BlockerType, ...] = (
    "PREMISE_INVALIDATED",
    "INCONSISTENT_REQUIREMENT",
    "MISSING_CONTRACT",
)
VALID_STATUSES: tuple[WorkerStatus, ...] = (
    "idle",
    "running",
    "completed",
    "preempted",
    "challenged",
)
VALID_DECISIONS: tuple[LeadDecision, ...] = ("replan", "override")
VALID_OUTCOMES: tuple[AttributionOutcome, ...] = (
    "compliant_flow",
    "valid_whistle_replan",
    "valid_whistle_override_fail",
    "fake_challenge_hack",
    "tamper_hack",
    "pingpong_exceeded",
    "protocol_error",
)


@dataclass(frozen=True)
class ArenaSession:
    """Persistent serializable match state (never stores credentials or hidden tests)."""

    task_id: str
    mode: OrchestrationMode
    phase: ArenaPhase
    human_actors: tuple[str, ...]
    worker_queue: tuple[str, ...]
    pending_challenge: ChallengePayload | None
    state: EpisodeState
    workspace_files: Mapping[str, str]
    enforce_write_lock: bool = True
    model: str = "deepseek-flash"
    base_url: str | None = None
    log_path: str | None = None
    max_protocol_retries: int = 2
    max_steps_override: int | None = None
    max_challenges_override: int | None = None
    verdict: JudgeVerdict | None = None
    rewards: RewardBreakdown | None = None


def _req_mapping(val: object, ctx: str) -> Mapping[str, object]:
    if not isinstance(val, Mapping):
        raise TypeError(f"Expected JSON object at '{ctx}'.")
    return val


def _req_list(val: object, ctx: str) -> list[object]:
    if not isinstance(val, list):
        raise TypeError(f"Expected JSON list at '{ctx}'.")
    return val


def _req_str(val: object, ctx: str) -> str:
    if not isinstance(val, str):
        raise TypeError(f"Expected string at '{ctx}'.")
    return val


def _req_int(val: object, ctx: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"Expected int at '{ctx}'.")
    return val


def _req_float(val: object, ctx: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"Expected float at '{ctx}'.")
    return float(val)


def _req_bool(val: object, ctx: str) -> bool:
    if not isinstance(val, bool):
        raise TypeError(f"Expected bool at '{ctx}'.")
    return val


def _req_literal(val: object, allowed: tuple[str, ...], ctx: str) -> str:
    s = _req_str(val, ctx)
    if s not in allowed:
        raise ValueError(f"Invalid value '{s}' at '{ctx}'; expected one of {allowed}.")
    return s


def _opt_str(val: object, ctx: str) -> str | None:
    if val is None:
        return None
    return _req_str(val, ctx)


def _opt_int(val: object, ctx: str) -> int | None:
    if val is None:
        return None
    return _req_int(val, ctx)


def _str_tuple(val: object, ctx: str) -> tuple[str, ...]:
    return tuple(_req_str(item, f"{ctx}[{i}]") for i, item in enumerate(_req_list(val, ctx)))


def _challenge_to_dict(ch: ChallengePayload) -> dict[str, object]:
    return {
        "worker_id": ch.worker_id,
        "step": ch.step,
        "blocker_type": ch.blocker_type,
        "invalidated_premise": ch.invalidated_premise,
        "evidence_file": ch.evidence_file,
        "evidence_excerpt": ch.evidence_excerpt,
        "proposed_hint": ch.proposed_hint,
    }


def _challenge_from_dict(raw: object, ctx: str) -> ChallengePayload:
    obj = _req_mapping(raw, ctx)
    blocker = _req_literal(obj.get("blocker_type"), VALID_BLOCKERS, f"{ctx}.blocker_type")
    return ChallengePayload(
        worker_id=_req_str(obj.get("worker_id"), f"{ctx}.worker_id"),
        step=_req_int(obj.get("step"), f"{ctx}.step"),
        blocker_type=cast(BlockerType, blocker),
        invalidated_premise=_req_str(obj.get("invalidated_premise"), f"{ctx}.invalidated_premise"),
        evidence_file=_req_str(obj.get("evidence_file"), f"{ctx}.evidence_file"),
        evidence_excerpt=_req_str(obj.get("evidence_excerpt"), f"{ctx}.evidence_excerpt"),
        proposed_hint=_req_str(obj.get("proposed_hint", ""), f"{ctx}.proposed_hint"),
    )


def _assignment_to_dict(a: WorkerAssignment) -> dict[str, object]:
    return {
        "worker_id": a.worker_id,
        "goal": a.goal,
        "explicit_premises": list(a.explicit_premises),
        "generation": a.generation,
        "writable_paths": list(a.writable_paths),
    }


def _assignment_from_dict(raw: object, ctx: str) -> WorkerAssignment:
    obj = _req_mapping(raw, ctx)
    return WorkerAssignment(
        worker_id=_req_str(obj.get("worker_id"), f"{ctx}.worker_id"),
        goal=_req_str(obj.get("goal"), f"{ctx}.goal"),
        explicit_premises=_str_tuple(obj.get("explicit_premises"), f"{ctx}.explicit_premises"),
        generation=_req_int(obj.get("generation", 0), f"{ctx}.generation"),
        writable_paths=_str_tuple(obj.get("writable_paths", []), f"{ctx}.writable_paths"),
    )


def _worker_to_dict(w: WorkerState) -> dict[str, object]:
    return {
        "worker_id": w.worker_id,
        "status": w.status,
        "assignment": _assignment_to_dict(w.assignment),
        "local_notes": list(w.local_notes),
        "steps_taken": w.steps_taken,
        "respawn_count": w.respawn_count,
    }


def _worker_from_dict(raw: object, ctx: str) -> WorkerState:
    obj = _req_mapping(raw, ctx)
    status = _req_literal(obj.get("status"), VALID_STATUSES, f"{ctx}.status")
    return WorkerState(
        worker_id=_req_str(obj.get("worker_id"), f"{ctx}.worker_id"),
        status=cast(WorkerStatus, status),
        assignment=_assignment_from_dict(obj.get("assignment"), f"{ctx}.assignment"),
        local_notes=_str_tuple(obj.get("local_notes", []), f"{ctx}.local_notes"),
        steps_taken=_req_int(obj.get("steps_taken", 0), f"{ctx}.steps_taken"),
        respawn_count=_req_int(obj.get("respawn_count", 0), f"{ctx}.respawn_count"),
    )


def _obs_to_dict(obs: ToolObservation) -> dict[str, object]:
    return {
        "step": obs.step,
        "worker_id": obs.worker_id,
        "tool_name": obs.tool_name,
        "target": obs.target,
        "exit_code": obs.exit_code,
        "excerpt": obs.excerpt,
        "content_sha256": obs.content_sha256,
    }


def _obs_from_dict(raw: object, ctx: str) -> ToolObservation:
    obj = _req_mapping(raw, ctx)
    return ToolObservation(
        step=_req_int(obj.get("step"), f"{ctx}.step"),
        worker_id=_req_str(obj.get("worker_id"), f"{ctx}.worker_id"),
        tool_name=_req_str(obj.get("tool_name"), f"{ctx}.tool_name"),
        target=_req_str(obj.get("target"), f"{ctx}.target"),
        exit_code=_req_int(obj.get("exit_code"), f"{ctx}.exit_code"),
        excerpt=_req_str(obj.get("excerpt"), f"{ctx}.excerpt"),
        content_sha256=_req_str(obj.get("content_sha256"), f"{ctx}.content_sha256"),
    )


def _event_to_dict(ev: EventLogItem) -> dict[str, object]:
    return {
        "step": ev.step,
        "actor": ev.actor,
        "action": ev.action,
        "detail": ev.detail,
        "depends_on_steps": list(ev.depends_on_steps),
        "cost_units": ev.cost_units,
    }


def _event_from_dict(raw: object, ctx: str) -> EventLogItem:
    obj = _req_mapping(raw, ctx)
    deps = tuple(
        _req_int(d, f"{ctx}.depends_on_steps[{i}]")
        for i, d in enumerate(_req_list(obj.get("depends_on_steps", []), f"{ctx}.depends_on_steps"))
    )
    return EventLogItem(
        step=_req_int(obj.get("step"), f"{ctx}.step"),
        actor=_req_str(obj.get("actor"), f"{ctx}.actor"),
        action=_req_str(obj.get("action"), f"{ctx}.action"),
        detail=_req_str(obj.get("detail"), f"{ctx}.detail"),
        depends_on_steps=deps,
        cost_units=_req_float(obj.get("cost_units", 1.0), f"{ctx}.cost_units"),
    )


def _state_to_dict(state: EpisodeState) -> dict[str, object]:
    return {
        "task_id": state.task_id,
        "mode": state.mode,
        "step": state.step,
        "workers": {wid: _worker_to_dict(w) for wid, w in state.workers.items()},
        "observations": [_obs_to_dict(o) for o in state.observations],
        "challenges": [_challenge_to_dict(c) for c in state.challenges],
        "lead_decisions": [list(d) for d in state.lead_decisions],
        "events": [_event_to_dict(e) for e in state.events],
        "public_tests_passed": state.public_tests_passed,
        "terminated": state.terminated,
        "termination_reason": state.termination_reason,
    }


def _decision_from_list(raw: object, ctx: str) -> tuple[int, LeadDecision, str]:
    item = _req_list(raw, ctx)
    if len(item) != 3:
        raise ValueError(f"Expected 3-element [step, decision, reason] at '{ctx}'.")
    dec = _req_literal(item[1], VALID_DECISIONS, f"{ctx}[1]")
    return (_req_int(item[0], f"{ctx}[0]"), cast(LeadDecision, dec), _req_str(item[2], f"{ctx}[2]"))


def _state_from_dict(raw: object, ctx: str) -> EpisodeState:
    obj = _req_mapping(raw, ctx)
    mode = _req_literal(obj.get("mode"), VALID_MODES, f"{ctx}.mode")
    raw_workers = _req_mapping(obj.get("workers", {}), f"{ctx}.workers")
    workers = {
        _req_str(k, f"{ctx}.workers.key"): _worker_from_dict(v, f"{ctx}.workers[{k}]")
        for k, v in raw_workers.items()
    }
    observations = tuple(
        _obs_from_dict(o, f"{ctx}.observations[{i}]")
        for i, o in enumerate(_req_list(obj.get("observations", []), f"{ctx}.observations"))
    )
    challenges = tuple(
        _challenge_from_dict(c, f"{ctx}.challenges[{i}]")
        for i, c in enumerate(_req_list(obj.get("challenges", []), f"{ctx}.challenges"))
    )
    lead_decisions = tuple(
        _decision_from_list(raw_d, f"{ctx}.lead_decisions[{i}]")
        for i, raw_d in enumerate(_req_list(obj.get("lead_decisions", []), f"{ctx}.lead_decisions"))
    )
    events = tuple(
        _event_from_dict(e, f"{ctx}.events[{i}]")
        for i, e in enumerate(_req_list(obj.get("events", []), f"{ctx}.events"))
    )
    return EpisodeState(
        task_id=_req_str(obj.get("task_id"), f"{ctx}.task_id"),
        mode=cast(OrchestrationMode, mode),
        step=_req_int(obj.get("step"), f"{ctx}.step"),
        workers=freeze_mapping_generic(workers),
        observations=observations,
        challenges=challenges,
        lead_decisions=lead_decisions,
        events=events,
        public_tests_passed=_req_bool(obj.get("public_tests_passed", False), f"{ctx}.public_tests_passed"),
        terminated=_req_bool(obj.get("terminated", False), f"{ctx}.terminated"),
        termination_reason=_req_str(obj.get("termination_reason", ""), f"{ctx}.termination_reason"),
    )


def _verdict_to_dict(v: JudgeVerdict) -> dict[str, object]:
    return {
        "task_id": v.task_id,
        "sandbox_intact": v.sandbox_intact,
        "tampered_files": list(v.tampered_files),
        "public_passed": v.public_passed,
        "hidden_passed": v.hidden_passed,
        "hidden_summary": v.hidden_summary,
        "challenge_audits": [
            {
                "worker_id": a.worker_id,
                "step": a.step,
                "evidence_grounded": a.evidence_grounded,
                "conflict_verified": a.conflict_verified,
                "reason": a.reason,
            }
            for a in v.challenge_audits
        ],
        "attribution_outcome": v.attribution_outcome,
        "critical_path_cost": v.critical_path_cost,
    }


def _verdict_from_dict(raw: object, ctx: str) -> JudgeVerdict:
    obj = _req_mapping(raw, ctx)
    audits = tuple(
        ChallengeAudit(
            worker_id=_req_str(m.get("worker_id"), f"{ctx}.challenge_audits[{i}].worker_id"),
            step=_req_int(m.get("step"), f"{ctx}.challenge_audits[{i}].step"),
            evidence_grounded=_req_bool(
                m.get("evidence_grounded"), f"{ctx}.challenge_audits[{i}].evidence_grounded"
            ),
            conflict_verified=_req_bool(
                m.get("conflict_verified"), f"{ctx}.challenge_audits[{i}].conflict_verified"
            ),
            reason=_req_str(m.get("reason"), f"{ctx}.challenge_audits[{i}].reason"),
        )
        for i, item in enumerate(
            _req_list(obj.get("challenge_audits", []), f"{ctx}.challenge_audits")
        )
        for m in [_req_mapping(item, f"{ctx}.challenge_audits[{i}]")]
    )
    return JudgeVerdict(
        task_id=_req_str(obj.get("task_id"), f"{ctx}.task_id"),
        sandbox_intact=_req_bool(obj.get("sandbox_intact"), f"{ctx}.sandbox_intact"),
        tampered_files=_str_tuple(obj.get("tampered_files", []), f"{ctx}.tampered_files"),
        public_passed=_req_bool(obj.get("public_passed"), f"{ctx}.public_passed"),
        hidden_passed=_req_bool(obj.get("hidden_passed"), f"{ctx}.hidden_passed"),
        hidden_summary=_req_str(obj.get("hidden_summary"), f"{ctx}.hidden_summary"),
        challenge_audits=audits,
        attribution_outcome=cast(
            AttributionOutcome,
            _req_literal(obj.get("attribution_outcome"), VALID_OUTCOMES, f"{ctx}.attribution_outcome"),
        ),
        critical_path_cost=_req_float(obj.get("critical_path_cost"), f"{ctx}.critical_path_cost"),
    )


def _rewards_to_dict(r: RewardBreakdown) -> dict[str, object]:
    return {
        "lead_reward": r.lead_reward,
        "subagent_reward": r.subagent_reward,
        "outcome_label": r.outcome_label,
        "notes": list(r.notes),
        "per_worker_rewards": dict(r.per_worker_rewards),
    }


def _rewards_from_dict(raw: object, ctx: str) -> RewardBreakdown:
    obj = _req_mapping(raw, ctx)
    raw_pw = _req_mapping(obj.get("per_worker_rewards", {}), f"{ctx}.per_worker_rewards")
    per_worker = MappingProxyType(
        {
            _req_str(k, f"{ctx}.per_worker_rewards.key"): _req_float(
                v, f"{ctx}.per_worker_rewards[{k}]"
            )
            for k, v in raw_pw.items()
        }
    )
    return RewardBreakdown(
        lead_reward=_req_float(obj.get("lead_reward"), f"{ctx}.lead_reward"),
        subagent_reward=_req_float(obj.get("subagent_reward"), f"{ctx}.subagent_reward"),
        outcome_label=cast(
            AttributionOutcome,
            _req_literal(obj.get("outcome_label"), VALID_OUTCOMES, f"{ctx}.outcome_label"),
        ),
        notes=_str_tuple(obj.get("notes", []), f"{ctx}.notes"),
        per_worker_rewards=per_worker,
    )


def arena_to_dict(session: ArenaSession) -> dict[str, object]:
    """Serialize an immutable ArenaSession to a plain JSON-safe dict without secrets."""
    return {
        "schema_version": ARENA_SCHEMA_VERSION,
        "task_id": session.task_id,
        "mode": session.mode,
        "phase": session.phase,
        "human_actors": list(session.human_actors),
        "worker_queue": list(session.worker_queue),
        "pending_challenge": (
            _challenge_to_dict(session.pending_challenge) if session.pending_challenge else None
        ),
        "state": _state_to_dict(session.state),
        "workspace_files": {
            k: v for k, v in session.workspace_files.items() if not k.startswith("hidden_tests/")
        },
        "enforce_write_lock": session.enforce_write_lock,
        "model": session.model,
        "base_url": session.base_url,
        "log_path": session.log_path,
        "max_protocol_retries": session.max_protocol_retries,
        "max_steps_override": session.max_steps_override,
        "max_challenges_override": session.max_challenges_override,
        "verdict": _verdict_to_dict(session.verdict) if session.verdict else None,
        "rewards": _rewards_to_dict(session.rewards) if session.rewards else None,
    }


def arena_from_dict(raw: Mapping[str, object]) -> ArenaSession:
    """Deserialize and strictly validate an ArenaSession from a parsed JSON mapping."""
    ver = _req_int(raw.get("schema_version"), "schema_version")
    if ver != ARENA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported arena schema_version={ver} (expected {ARENA_SCHEMA_VERSION}).")
    mode = _req_str(raw.get("mode"), "mode")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'.")
    phase = _req_str(raw.get("phase"), "phase")
    if phase not in VALID_PHASES:
        raise ValueError(f"Unknown phase '{phase}'.")
    raw_ws = _req_mapping(raw.get("workspace_files"), "workspace_files")
    ws = {
        _req_str(k, "workspace_files.key"): _req_str(v, f"workspace_files[{k}]")
        for k, v in raw_ws.items()
        if not str(k).startswith("hidden_tests/")
    }
    raw_ch = raw.get("pending_challenge")
    raw_verdict = raw.get("verdict")
    raw_rewards = raw.get("rewards")
    return ArenaSession(
        task_id=_req_str(raw.get("task_id"), "task_id"),
        mode=cast(OrchestrationMode, mode),
        phase=cast(ArenaPhase, phase),
        human_actors=_str_tuple(raw.get("human_actors", []), "human_actors"),
        worker_queue=_str_tuple(raw.get("worker_queue", []), "worker_queue"),
        pending_challenge=_challenge_from_dict(raw_ch, "pending_challenge") if raw_ch else None,
        state=_state_from_dict(raw.get("state"), "state"),
        workspace_files=freeze_mapping(ws),
        enforce_write_lock=_req_bool(raw.get("enforce_write_lock", True), "enforce_write_lock"),
        model=_req_str(raw.get("model", "deepseek-flash"), "model"),
        base_url=_opt_str(raw.get("base_url"), "base_url"),
        log_path=_opt_str(raw.get("log_path"), "log_path"),
        max_protocol_retries=_req_int(raw.get("max_protocol_retries", 2), "max_protocol_retries"),
        max_steps_override=_opt_int(raw.get("max_steps_override"), "max_steps_override"),
        max_challenges_override=_opt_int(
            raw.get("max_challenges_override"), "max_challenges_override"
        ),
        verdict=_verdict_from_dict(raw_verdict, "verdict") if raw_verdict else None,
        rewards=_rewards_from_dict(raw_rewards, "rewards") if raw_rewards else None,
    )


def save_arena(path: Path, session: ArenaSession) -> None:
    """Atomically write an ArenaSession JSON checkpoint without touching existing target on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(arena_to_dict(session), ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_arena(path: Path) -> ArenaSession:
    """Load and validate an ArenaSession JSON checkpoint from disk."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return arena_from_dict(_req_mapping(parsed, str(path)))


def _apply_task_overrides(
    base: TaskSpec,
    max_steps: int | None,
    max_challenges: int | None,
) -> TaskSpec:
    resolved_steps = base.max_steps if max_steps is None else max_steps
    resolved_challenges = base.max_challenges if max_challenges is None else max_challenges
    if resolved_steps == base.max_steps and resolved_challenges == base.max_challenges:
        return base
    return replace(base, max_steps=resolved_steps, max_challenges=resolved_challenges)


def effective_task(session: ArenaSession) -> TaskSpec:
    """Resolve TaskSpec with any session-level step/challenge budget overrides applied."""
    return _apply_task_overrides(
        get_task_by_id(session.task_id),
        session.max_steps_override,
        session.max_challenges_override,
    )


def is_human_actor(actor: str, human_actors: tuple[str, ...]) -> bool:
    """Return True when the given actor ID ('lead', 'w1', ...) is human-controlled."""
    return actor in human_actors or (actor != "lead" and "workers" in human_actors)


def next_arena_actor(session: ArenaSession) -> str | None:
    """Return the next actor scheduled to act, or None if the match is finished."""
    if session.phase == "finished":
        return None
    if session.phase in ("awaiting_lead_init", "awaiting_lead_response"):
        return "lead"
    return session.worker_queue[0] if session.worker_queue else None


def _refill_worker_queue(state: EpisodeState, candidate_tail: tuple[str, ...]) -> tuple[str, ...]:
    filtered = tuple(
        wid
        for wid in candidate_tail
        if wid in state.workers and state.workers[wid].status == "running"
    )
    if filtered:
        return filtered
    return tuple(w.worker_id for w in state.workers.values() if w.status == "running")


def _finalize_episode(
    session: ArenaSession,
    task: TaskSpec,
    state: EpisodeState,
    workspace_files: Mapping[str, str],
    reason: str | None = None,
) -> ArenaSession:
    resolved_reason = reason or state.termination_reason
    sandbox = SandboxSession(task, initial_workspace=workspace_files)
    try:
        public_final = sandbox.run_public_tests(state.step + 1, "harness")
        final_ws = sandbox.snapshot_all()
    finally:
        sandbox.close()
    final_state = replace(
        state,
        public_tests_passed=(public_final.exit_code == 0),
        terminated=True,
        termination_reason=resolved_reason,
    )
    verdict = evaluate_episode(task, final_state, final_ws)
    rewards = compute_rewards(verdict, final_state)
    return replace(
        session,
        phase="finished",
        worker_queue=(),
        pending_challenge=None,
        state=final_state,
        workspace_files=final_ws,
        verdict=verdict,
        rewards=rewards,
    )


def _settle_post_decision(
    session: ArenaSession,
    task: TaskSpec,
    state: EpisodeState,
    workspace_files: Mapping[str, str],
    candidate_tail: tuple[str, ...],
) -> ArenaSession:
    if state.terminated:
        return _finalize_episode(session, task, state, workspace_files)
    queue = _refill_worker_queue(state, candidate_tail)
    if not queue:
        return _finalize_episode(session, task, state, workspace_files, reason="all_workers_done")
    if state.step >= task.max_steps:
        return _finalize_episode(session, task, state, workspace_files, reason="max_steps_reached")
    return replace(
        session,
        phase="awaiting_worker",
        worker_queue=queue,
        pending_challenge=None,
        state=state,
        workspace_files=workspace_files,
    )


def create_arena_session(
    task_id: str,
    mode: OrchestrationMode,
    *,
    human_actors: tuple[str, ...] = (),
    enforce_write_lock: bool = True,
    model: str = "deepseek-flash",
    base_url: str | None = None,
    log_path: str | None = None,
    max_protocol_retries: int = 2,
    max_steps: int | None = None,
    max_challenges: int | None = None,
) -> ArenaSession:
    """Create a fresh ArenaSession initialized with Lead scout observations."""
    task = _apply_task_overrides(get_task_by_id(task_id), max_steps, max_challenges)

    sandbox = SandboxSession(task)
    try:
        scout_obs = collect_scout_observations(task, mode, sandbox)
        ws = sandbox.snapshot_all()
    finally:
        sandbox.close()

    scout_state = create_scout_state(task, mode, scout_obs)
    return ArenaSession(
        task_id=task.task_id,
        mode=mode,
        phase="awaiting_lead_init",
        human_actors=tuple(dict.fromkeys(human_actors)),
        worker_queue=(),
        pending_challenge=None,
        state=scout_state,
        workspace_files=ws,
        enforce_write_lock=enforce_write_lock,
        model=model,
        base_url=base_url,
        log_path=log_path,
        max_protocol_retries=max_protocol_retries,
        max_steps_override=max_steps,
        max_challenges_override=max_challenges,
    )


def render_turn_prompt(session: ArenaSession) -> dict[str, object] | None:
    """Build the role-isolated turn prompt & schema for whichever actor moves next."""
    if session.phase == "finished":
        return None
    task = effective_task(session)
    if session.phase == "awaiting_lead_init":
        sys_p, user_p = build_lead_init_prompt(task, session.mode, session.state.observations)
        return {
            "actor": "lead",
            "role": "lead_init",
            "phase": session.phase,
            "next_step": session.state.step + 1,
            "system": sys_p,
            "user": user_p,
        }
    if session.phase == "awaiting_worker":
        wid = session.worker_queue[0]
        sys_p, user_p = build_worker_action_prompt(task, session.state, session.state.workers[wid])
        return {
            "actor": wid,
            "role": "worker",
            "phase": session.phase,
            "next_step": session.state.step + 1,
            "system": sys_p,
            "user": user_p,
        }
    if session.pending_challenge is None:
        raise ValueError("Arena phase is 'awaiting_lead_response' but pending_challenge is None.")
    sys_p, user_p = build_lead_response_prompt(task, session.state, session.pending_challenge)
    return {
        "actor": "lead",
        "role": "lead_respond",
        "phase": session.phase,
        "next_step": session.state.step + 1,
        "system": sys_p,
        "user": user_p,
    }


def _apply_worker_parsed_action(
    session: ArenaSession,
    task: TaskSpec,
    worker: WorkerState,
    action: WorkerAction,
    step: int,
) -> ArenaSession:
    sandbox = SandboxSession(task, initial_workspace=session.workspace_files)
    try:
        next_state, issued_challenge = execute_worker_turn_step(
            session.state,
            task,
            sandbox,
            worker,
            action,
            step,
            enforce_write_lock=session.enforce_write_lock,
        )
        next_ws = (
            sandbox.snapshot_all()
            if isinstance(action, ToolAction) and action.tool_name == "write_file"
            else session.workspace_files
        )
    finally:
        sandbox.close()

    if issued_challenge is not None:
        if next_state.terminated:
            return _finalize_episode(session, task, next_state, next_ws)
        return replace(
            session,
            phase="awaiting_lead_response",
            worker_queue=(),
            pending_challenge=issued_challenge,
            state=next_state,
            workspace_files=next_ws,
        )
    return _settle_post_decision(session, task, next_state, next_ws, session.worker_queue[1:])


def _append_human_log(
    session: ArenaSession,
    *,
    step: int,
    role: str,
    actor: str,
    raw_json: str,
    summary: str,
) -> None:
    if session.log_path is None:
        return
    append_jsonl_log(
        Path(session.log_path),
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "task_id": session.task_id,
            "mode": session.mode,
            "step": step,
            "role": role,
            "actor": actor,
            "attempt": 1,
            "model": "human",
            "parsed_ok": True,
            "parsed_summary": summary,
            "raw_response": raw_json,
        },
    )


def step_human_turn(session: ArenaSession, raw_json: str) -> ArenaSession:
    """Validate and apply one human JSON turn cleanly without mutating session on invalid input."""
    actor = next_arena_actor(session)
    if actor is None:
        raise ValueError("Match is already finished.")
    if not is_human_actor(actor, session.human_actors):
        raise ValueError(
            f"Next turn belongs to LLM actor '{actor}', not human (human_actors={session.human_actors})."
        )
    task = effective_task(session)
    next_step = session.state.step + 1

    if session.phase == "awaiting_lead_init":
        assignments = parse_assignments(raw_json)
        _append_human_log(
            session,
            step=next_step,
            role="lead_init",
            actor="lead",
            raw_json=raw_json,
            summary=f"assignments({[(a.worker_id, a.writable_paths) for a in assignments]})",
        )
        next_state = apply_initial_assignments(session.state, assignments)
        return _settle_post_decision(session, task, next_state, session.workspace_files, ())

    if session.phase == "awaiting_worker":
        worker = session.state.workers[actor]
        action = parse_worker_action(raw_json, actor, next_step, session.mode)
        _append_human_log(
            session,
            step=next_step,
            role="worker",
            actor=actor,
            raw_json=raw_json,
            summary=type(action).__name__,
        )
        return _apply_worker_parsed_action(session, task, worker, action, next_step)

    if session.pending_challenge is None:
        raise ValueError("Missing pending_challenge while awaiting lead response.")
    resp: LeadResponse = parse_lead_response(raw_json, session.state)
    _append_human_log(
        session,
        step=next_step,
        role="lead_respond",
        actor="lead",
        raw_json=raw_json,
        summary=f"lead:{resp.decision}(workers={len(resp.new_assignments)})",
    )
    next_state = apply_lead_response(
        session.state, resp, session.pending_challenge.worker_id, next_step
    )
    return _settle_post_decision(session, task, next_state, session.workspace_files, ())


def step_llm_turn(
    session: ArenaSession,
    team: LiveLLMPolicyTeam | None = None,
) -> ArenaSession:
    """Execute one LLM turn (using LiveLLMPolicyTeam) or catch protocol exhaustion cleanly."""
    actor = next_arena_actor(session)
    if actor is None:
        raise ValueError("Match is already finished.")
    if is_human_actor(actor, session.human_actors):
        raise ValueError(
            f"Next turn belongs to human actor '{actor}'; submit --action-json or --action-file."
        )
    task = effective_task(session)
    resolved_team = team or LiveLLMPolicyTeam.from_env(
        model=session.model,
        base_url=session.base_url,
        log_path=Path(session.log_path) if session.log_path else None,
        max_protocol_retries=session.max_protocol_retries,
    )
    next_step = session.state.step + 1

    try:
        if session.phase == "awaiting_lead_init":
            assignments = resolved_team.choose_initial_assignments(
                task, session.mode, session.state.observations
            )
            next_state = apply_initial_assignments(session.state, assignments)
            return _settle_post_decision(session, task, next_state, session.workspace_files, ())

        if session.phase == "awaiting_worker":
            worker = session.state.workers[actor]
            action = resolved_team.next_worker_action(task, session.state, worker)
            return _apply_worker_parsed_action(session, task, worker, action, next_step)

        if session.pending_challenge is None:
            raise ValueError("Missing pending_challenge while awaiting lead response.")
        resp = resolved_team.respond_to_challenge(task, session.state, session.pending_challenge)
        next_state = apply_lead_response(
            session.state, resp, session.pending_challenge.worker_id, next_step
        )
        return _settle_post_decision(session, task, next_state, session.workspace_files, ())
    except ProtocolViolationError as err:
        err_state = terminate_protocol_error(session.state, err, step=next_step)
        return _finalize_episode(session, task, err_state, session.workspace_files)


def advance_arena(
    session: ArenaSession,
    *,
    max_turns: int = 30,
    team: LiveLLMPolicyTeam | None = None,
) -> tuple[ArenaSession, tuple[dict[str, object], ...]]:
    """Run consecutive LLM turns until the next human turn, finish, or max_turns cap."""
    current = session
    trace: list[dict[str, object]] = []
    for _ in range(max_turns):
        actor = next_arena_actor(current)
        if actor is None or is_human_actor(actor, current.human_actors):
            break
        prev_len = len(current.state.events)
        current = step_llm_turn(current, team=team)
        for ev in current.state.events[prev_len:]:
            trace.append(
                {
                    "step": ev.step,
                    "actor": ev.actor,
                    "action": ev.action,
                    "detail": ev.detail[:140],
                }
            )
    return current, tuple(trace)


def arena_summary_dict(session: ArenaSession) -> dict[str, object]:
    """Produce a spoiler-safe summary of current match progress (grade unlocked on finish)."""
    task = effective_task(session)
    canonical = {**task.initial_files, **task.public_test_files}
    modified = tuple(
        sorted(
            path
            for path in set(canonical.keys()) | set(session.workspace_files.keys())
            if canonical.get(path) != session.workspace_files.get(path)
        )
    )
    summary: dict[str, object] = {
        "task_id": task.task_id,
        "title": task.title,
        "mode": session.mode,
        "phase": session.phase,
        "next_actor": next_arena_actor(session),
        "human_actors": list(session.human_actors),
        "step": session.state.step,
        "max_steps": task.max_steps,
        "challenges_used": len(session.state.challenges),
        "max_challenges": task.max_challenges,
        "workers": {
            wid: {
                "status": w.status,
                "generation": w.assignment.generation,
                "steps_taken": w.steps_taken,
                "goal": w.assignment.goal,
                "writable_paths": list(w.assignment.writable_paths),
            }
            for wid, w in session.state.workers.items()
        },
        "pending_challenge": (
            _challenge_to_dict(session.pending_challenge) if session.pending_challenge else None
        ),
        "modified_files": list(modified),
        "recent_events": [
            f"s{e.step}:{e.actor}:{e.action}({e.detail[:100]})"
            for e in session.state.events[-8:]
        ],
    }
    if session.phase == "finished" and session.verdict and session.rewards:
        summary["termination_reason"] = session.state.termination_reason
        summary["verdict"] = _verdict_to_dict(session.verdict)
        summary["rewards"] = _rewards_to_dict(session.rewards)
    return summary


def format_arena_status(session: ArenaSession) -> str:
    """Format a concise human-readable status report for CLI output."""
    info = arena_summary_dict(session)
    lines = [
        f"Match: `{info['task_id']}` ({info['mode']}) | phase=`{info['phase']}` | next=`{info['next_actor']}`",
        f"Step: {info['step']}/{info['max_steps']} | Challenges: {info['challenges_used']}/{info['max_challenges']} | Modified: {info['modified_files']}",
    ]
    workers = info["workers"]
    if isinstance(workers, dict) and workers:
        for wid, winfo in workers.items():
            lines.append(
                f"  - {wid} [{winfo['status']}] gen={winfo['generation']} "
                f"writable={winfo['writable_paths']} | {winfo['goal']}"
            )
    if session.pending_challenge:
        ch = session.pending_challenge
        lines.append(
            f"Pending challenge from {ch.worker_id} [{ch.blocker_type} on {ch.evidence_file}]: {ch.invalidated_premise}"
        )
    recent = info["recent_events"]
    if isinstance(recent, list) and recent:
        lines.append("Recent events: " + " -> ".join(str(item) for item in recent))
    if session.phase == "finished" and session.verdict and session.rewards:
        v = session.verdict
        r = session.rewards
        lines.append(
            f"FINAL RESULT: public={v.public_passed} hidden={v.hidden_passed} intact={v.sandbox_intact} "
            f"outcome=`{r.outcome_label}` R_lead={r.lead_reward:+.2f} R_sub={r.subagent_reward:+.2f} "
            f"per_worker={dict(r.per_worker_rewards)}"
        )
    return "\n".join(lines)


def _read_action_text(action_json: str | None, action_file: str | None) -> str | None:
    if action_json is not None and action_file is not None:
        raise ValueError("Pass only one of --action-json or --action-file.")
    if action_json is not None:
        return sys.stdin.read() if action_json == "-" else action_json
    if action_file is not None:
        return sys.stdin.read() if action_file == "-" else Path(action_file).read_text(encoding="utf-8")
    return None


def _emit_turn_output(
    session: ArenaSession,
    *,
    as_json: bool,
    executed: tuple[dict[str, object], ...] = (),
) -> None:
    if as_json:
        payload: dict[str, object] = {
            "summary": arena_summary_dict(session),
            "next_prompt": render_turn_prompt(session),
        }
        if executed:
            payload["executed_llm_turns"] = list(executed)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if executed:
        print("LLM turns executed:")
        for item in executed:
            print(f"  s{item['step']} [{item['actor']}] {item['action']}: {item['detail']}")
    print(format_arena_status(session))


def main(argv: list[str] | None = None) -> int:
    """Run Interactive Arena subcommands (`tasks`, `init`, `status`, `prompt`, `step`, `advance`)."""
    parser = argparse.ArgumentParser(
        prog="hidden_premise_lab.arena",
        description="Interactive turn-based Human vs LLM Arena with persistent JSON checkpoints",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tasks = sub.add_parser("tasks", help="List available benchmark tasks")
    p_tasks.add_argument("--json", action="store_true")

    p_init = sub.add_parser("init", help="Initialize a new match checkpoint JSON")
    p_init.add_argument("--task", required=True)
    p_init.add_argument("--mode", choices=VALID_MODES, default="bidirectional_escalation")
    p_init.add_argument("--state", required=True, help="Path to saved match JSON")
    p_init.add_argument("--human", action="append", default=[], help="Human actor IDs: lead, w1, w2, workers")
    p_init.add_argument("--model", default="deepseek-flash")
    p_init.add_argument("--base-url", default=None)
    p_init.add_argument("--log-file", default=None)
    p_init.add_argument("--max-protocol-retries", type=int, default=2)
    p_init.add_argument("--max-steps", type=int, default=None)
    p_init.add_argument("--max-challenges", type=int, default=None)
    p_init.add_argument("--no-write-lock", action="store_true")
    p_init.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="Inspect current match checkpoint")
    p_status.add_argument("--state", required=True)
    p_status.add_argument("--json", action="store_true")

    p_prompt = sub.add_parser("prompt", help="Print the exact next-actor prompt & schema")
    p_prompt.add_argument("--state", required=True)
    p_prompt.add_argument("--json", action="store_true")

    p_step = sub.add_parser("step", help="Advance one turn (human JSON or single LLM call)")
    p_step.add_argument("--state", required=True)
    p_step.add_argument("--action-json", default=None)
    p_step.add_argument("--action-file", default=None)
    p_step.add_argument("--until-human", action="store_true")
    p_step.add_argument("--max-turns", type=int, default=30)
    p_step.add_argument("--json", action="store_true")

    p_adv = sub.add_parser("advance", help="Run consecutive LLM turns until human turn or finish")
    p_adv.add_argument("--state", required=True)
    p_adv.add_argument("--max-turns", type=int, default=30)
    p_adv.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "tasks":
        items = [
            {
                "task_id": t.task_id,
                "title": t.title,
                "max_steps": t.max_steps,
                "max_challenges": t.max_challenges,
                "visible_files": sorted(set(t.initial_files.keys()) | set(t.public_test_files.keys())),
            }
            for t in all_benchmark_tasks()
        ]
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2))
        else:
            for item in items:
                print(f"- `{item['task_id']}`: {item['title']} (files={item['visible_files']})")
        return 0

    state_path = Path(args.state)
    if args.cmd == "init":
        session = create_arena_session(
            args.task,
            cast(OrchestrationMode, args.mode),
            human_actors=tuple(args.human),
            enforce_write_lock=not args.no_write_lock,
            model=args.model,
            base_url=args.base_url,
            log_path=args.log_file,
            max_protocol_retries=args.max_protocol_retries,
            max_steps=args.max_steps,
            max_challenges=args.max_challenges,
        )
        save_arena(state_path, session)
        _emit_turn_output(session, as_json=args.json)
        return 0

    session = load_arena(state_path)
    if args.cmd == "status":
        _emit_turn_output(session, as_json=args.json)
        return 0

    if args.cmd == "prompt":
        prompt = render_turn_prompt(session)
        if args.json:
            print(json.dumps(prompt, ensure_ascii=False, indent=2))
        elif prompt is None:
            print("Match finished; no further prompts.")
        else:
            print(f"=== ACTOR: {prompt['actor']} | ROLE: {prompt['role']} | STEP: {prompt['next_step']} ===")
            print(f"[SYSTEM]\n{prompt['system']}\n")
            print(f"[USER]\n{prompt['user']}")
        return 0

    if args.cmd == "step":
        action_text = _read_action_text(args.action_json, args.action_file)
        actor = next_arena_actor(session)
        if actor is None:
            _emit_turn_output(session, as_json=args.json)
            return 0
        try:
            if is_human_actor(actor, session.human_actors):
                if action_text is None:
                    raise ProtocolViolationError(
                        actor,
                        "MISSING_HUMAN_ACTION",
                        f"Next turn belongs to human actor '{actor}'; pass --action-json or --action-file.",
                    )
                updated = step_human_turn(session, action_text)
            else:
                if action_text is not None:
                    raise ValueError(f"Actor '{actor}' is LLM-controlled; remove --action-json/--action-file.")
                updated = step_llm_turn(session)
        except ProtocolViolationError as err:
            print(f"PROTOCOL_ERROR [{err.actor}] ({err.code}): {err.message}", file=sys.stderr)
            return 2

        executed: tuple[dict[str, object], ...] = ()
        if args.until_human and updated.phase != "finished":
            updated, executed = advance_arena(updated, max_turns=args.max_turns)
        save_arena(state_path, updated)
        _emit_turn_output(updated, as_json=args.json, executed=executed)
        return 0

    # args.cmd == "advance"
    updated, executed = advance_arena(session, max_turns=args.max_turns)
    save_arena(state_path, updated)
    _emit_turn_output(updated, as_json=args.json, executed=executed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
