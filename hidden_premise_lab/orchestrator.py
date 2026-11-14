"""Pure functional rollout state-machine supporting Fork-Join, Two-Stage Scout, and Bidirectional Escalation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, TypeVar

from hidden_premise_lab.models import (
    ChallengePayload,
    EpisodeState,
    EventLogItem,
    LeadDecision,
    OrchestrationMode,
    ProtocolViolationError,
    TaskSpec,
    ToolObservation,
    WorkerAssignment,
    WorkerState,
)
from hidden_premise_lab.sandbox import SandboxSession

V = TypeVar("V")


@dataclass(frozen=True)
class ToolAction:
    """Subagent action executing a sandbox tool."""

    worker_id: str
    tool_name: str
    arg1: str = ""
    arg2: str = ""


@dataclass(frozen=True)
class CompleteAction:
    """Subagent terminal completion report."""

    worker_id: str
    summary: str


@dataclass(frozen=True)
class ChallengeAction:
    """Subagent upward escalation challenging Lead's premise."""

    challenge: ChallengePayload


WorkerAction = ToolAction | CompleteAction | ChallengeAction


@dataclass(frozen=True)
class LeadResponse:
    """Lead's decision upon receiving an upward challenge."""

    decision: LeadDecision
    reason: str
    new_assignments: tuple[WorkerAssignment, ...]
    compressed_summary: str


class PolicyTeam(Protocol):
    """Protocol implemented by both scripted benchmark policies and live LLM runners."""

    def initial_plan(
        self, task: TaskSpec, mode: OrchestrationMode, sandbox: SandboxSession
    ) -> tuple[tuple[WorkerAssignment, ...], tuple[ToolObservation, ...]]: ...

    def next_worker_action(
        self, task: TaskSpec, state: EpisodeState, worker: WorkerState
    ) -> WorkerAction: ...

    def respond_to_challenge(
        self, task: TaskSpec, state: EpisodeState, challenge: ChallengePayload
    ) -> LeadResponse: ...


def create_scout_state(
    task: TaskSpec,
    mode: OrchestrationMode,
    scout_observations: tuple[ToolObservation, ...] = (),
) -> EpisodeState:
    """Create initial EpisodeState recording Lead scout reads prior to dispatching workers."""
    base_step = len(scout_observations)
    scout_events = tuple(
        EventLogItem(
            step=obs.step,
            actor="lead_scout",
            action=obs.tool_name,
            detail=obs.target,
            depends_on_steps=(obs.step - 1,) if obs.step > 1 else (),
            cost_units=1.0,
        )
        for obs in scout_observations
    )
    return EpisodeState(
        task_id=task.task_id,
        mode=mode,
        step=base_step,
        observations=scout_observations,
        events=scout_events,
    )


def apply_initial_assignments(
    state: EpisodeState,
    assignments: tuple[WorkerAssignment, ...],
) -> EpisodeState:
    """Attach initial Lead WorkerAssignments and dispatch DAG event onto a scout EpisodeState."""
    if not assignments:
        raise ValueError("Lead initial dispatch must provide non-empty assignments.")
    workers = {
        a.worker_id: WorkerState(worker_id=a.worker_id, status="running", assignment=a)
        for a in assignments
    }
    dispatch_step = state.step + 1
    deps = (state.step,) if state.step > 0 else ()
    dispatch_event = EventLogItem(
        step=dispatch_step,
        actor="lead",
        action="dispatch",
        detail=f"mode={state.mode}, workers={tuple(workers.keys())}",
        depends_on_steps=deps,
        cost_units=1.0,
    )
    return replace(
        state,
        step=dispatch_step,
        workers=freeze_mapping_generic(workers),
        events=state.events + (dispatch_event,),
    )


def init_episode_state(
    task: TaskSpec,
    mode: OrchestrationMode,
    assignments: tuple[WorkerAssignment, ...],
    scout_observations: tuple[ToolObservation, ...] = (),
) -> EpisodeState:
    """Create initial immutable EpisodeState with spawned workers and root DAG event."""
    return apply_initial_assignments(create_scout_state(task, mode, scout_observations), assignments)


def freeze_mapping_generic(data: Mapping[str, V]) -> Mapping[str, V]:
    """Wrap arbitrary dicts in MappingProxyType for immutability."""
    return MappingProxyType(dict(data))


def _last_actor_step(state: EpisodeState, actor: str) -> int:
    """Find causal parent step: max of actor's own last step and Lead's latest dispatch/decision."""
    actor_steps = tuple(e.step for e in state.events if e.actor == actor)
    lead_barrier = tuple(
        e.step
        for e in state.events
        if e.actor == "lead" and e.action in ("dispatch", "replan", "override")
    )
    return max(actor_steps + lead_barrier, default=1)


def apply_tool_observation(state: EpisodeState, obs: ToolObservation) -> EpisodeState:
    """Record a sandbox tool observation and advance the worker's DAG chain."""
    worker = state.workers[obs.worker_id]
    dep = _last_actor_step(state, obs.worker_id)
    updated_worker = replace(worker, steps_taken=worker.steps_taken + 1)
    updated_workers = freeze_mapping_generic({**state.workers, obs.worker_id: updated_worker})
    event = EventLogItem(
        step=obs.step,
        actor=obs.worker_id,
        action=obs.tool_name,
        detail=f"{obs.target} (rc={obs.exit_code})",
        depends_on_steps=(dep,),
        cost_units=1.0,
    )
    return replace(
        state,
        step=obs.step,
        workers=updated_workers,
        observations=state.observations + (obs,),
        events=state.events + (event,),
    )


def apply_worker_complete(state: EpisodeState, worker_id: str, summary: str, step: int) -> EpisodeState:
    """Mark a worker completed cleanly."""
    worker = state.workers[worker_id]
    dep = _last_actor_step(state, worker_id)
    updated_worker = replace(worker, status="completed", steps_taken=worker.steps_taken + 1)
    updated_workers = freeze_mapping_generic({**state.workers, worker_id: updated_worker})
    event = EventLogItem(
        step=step,
        actor=worker_id,
        action="complete",
        detail=summary,
        depends_on_steps=(dep,),
        cost_units=0.5,
    )
    return replace(state, step=step, workers=updated_workers, events=state.events + (event,))


def apply_challenge(
    state: EpisodeState,
    task: TaskSpec,
    challenge: ChallengePayload,
) -> EpisodeState:
    """Apply an upward challenge and preempt affected sibling workers (MPLMs B.2)."""
    dep = _last_actor_step(state, challenge.worker_id)
    new_challenges = state.challenges + (challenge,)
    over_limit = len(new_challenges) > task.max_challenges

    def transition_sibling(w: WorkerState) -> WorkerState:
        if w.worker_id == challenge.worker_id:
            return replace(w, status="challenged", steps_taken=w.steps_taken + 1)
        if w.status == "running":
            return replace(w, status="preempted")
        return w

    updated_workers = freeze_mapping_generic(
        {wid: transition_sibling(w) for wid, w in state.workers.items()}
    )
    event = EventLogItem(
        step=challenge.step,
        actor=challenge.worker_id,
        action="challenge",
        detail=f"{challenge.blocker_type} on {challenge.evidence_file}",
        depends_on_steps=(dep,),
        cost_units=1.0,
    )
    return replace(
        state,
        step=challenge.step,
        workers=updated_workers,
        challenges=new_challenges,
        events=state.events + (event,),
        terminated=over_limit,
        termination_reason="pingpong_limit" if over_limit else state.termination_reason,
    )


def apply_lead_response(
    state: EpisodeState,
    response: LeadResponse,
    challenger_id: str,
    step: int,
) -> EpisodeState:
    """Apply Lead's replan/respawn or override decision onto EpisodeState."""
    dep = _last_actor_step(state, challenger_id)
    decisions = state.lead_decisions + ((step, response.decision, response.reason),)

    if response.decision == "replan":
        if not response.new_assignments:
            raise ValueError("Lead replan must provide non-empty new_assignments.")
        next_workers = {
            a.worker_id: WorkerState(
                worker_id=a.worker_id,
                status="running",
                assignment=a,
                local_notes=(response.compressed_summary,),
                steps_taken=state.workers[a.worker_id].steps_taken if a.worker_id in state.workers else 0,
                respawn_count=(
                    state.workers[a.worker_id].respawn_count + 1
                    if a.worker_id in state.workers
                    else 0
                ),
            )
            for a in response.new_assignments
        }
    else:
        # Override keeps original premises, resumes workers, and delivers override reason to challenger/siblings.
        override_note = f"Lead override (step {step}, challenger={challenger_id}): {response.reason}"
        next_workers = {
            wid: (
                replace(w, status="running", local_notes=w.local_notes + (override_note,))
                if w.status in ("challenged", "preempted")
                else w
            )
            for wid, w in state.workers.items()
        }

    event = EventLogItem(
        step=step,
        actor="lead",
        action=response.decision,
        detail=response.reason,
        depends_on_steps=(dep,),
        cost_units=1.0,
    )
    return replace(
        state,
        step=step,
        workers=freeze_mapping_generic(next_workers),
        lead_decisions=decisions,
        events=state.events + (event,),
    )


def _execute_tool(
    sandbox: SandboxSession,
    step: int,
    action: ToolAction,
    worker: WorkerState,
    enforce_write_lock: bool = True,
) -> ToolObservation:
    if action.tool_name == "read_file":
        return sandbox.read_file(step, action.worker_id, action.arg1)
    if action.tool_name == "search_text":
        return sandbox.search_text(step, action.worker_id, action.arg1)
    if action.tool_name == "write_file":
        allowed = worker.assignment.writable_paths if enforce_write_lock else None
        return sandbox.write_file(step, action.worker_id, action.arg1, action.arg2, allowed_paths=allowed)
    if action.tool_name == "run_public_tests":
        return sandbox.run_public_tests(step, action.worker_id)
    raise ValueError(f"Unsupported tool: {action.tool_name}")


def terminate_protocol_error(
    state: EpisodeState,
    err: ProtocolViolationError,
    step: int,
) -> EpisodeState:
    """Record a terminal protocol_error DAG event without faking worker completion."""
    deps = (_last_actor_step(state, err.actor),) if state.events else ()
    event = EventLogItem(
        step=step,
        actor=err.actor,
        action="protocol_error",
        detail=f"{err.code}: {err.message}",
        depends_on_steps=deps,
        cost_units=1.0,
    )
    return replace(
        state,
        step=step,
        events=state.events + (event,),
        terminated=True,
        termination_reason="protocol_error",
    )


def execute_worker_turn_step(
    state: EpisodeState,
    task: TaskSpec,
    sandbox: SandboxSession,
    worker: WorkerState,
    action: WorkerAction,
    step: int,
    *,
    enforce_write_lock: bool = True,
) -> tuple[EpisodeState, ChallengePayload | None]:
    """Execute one parsed WorkerAction inside sandbox and return updated state + optional challenge."""
    if isinstance(action, ToolAction):
        obs = _execute_tool(
            sandbox,
            step,
            action,
            worker,
            enforce_write_lock=enforce_write_lock,
        )
        return apply_tool_observation(state, obs), None
    if isinstance(action, CompleteAction):
        return apply_worker_complete(state, worker.worker_id, action.summary, step), None
    if isinstance(action, ChallengeAction):
        if state.mode != "bidirectional_escalation":
            raise RuntimeError(f"Mode {state.mode} forbids upward challenge")
        fixed_challenge = replace(action.challenge, step=step)
        return apply_challenge(state, task, fixed_challenge), fixed_challenge
    raise TypeError(f"Unsupported worker action: {type(action).__name__}")


def run_episode(
    task: TaskSpec,
    mode: OrchestrationMode,
    policy: PolicyTeam,
    *,
    enforce_write_lock: bool = True,
) -> tuple[EpisodeState, dict[str, str]]:
    """Run one complete rollout inside an isolated temporary sandbox and return immutable snapshot."""
    sandbox = SandboxSession(task)
    try:
        try:
            assignments, scout_obs = policy.initial_plan(task, mode, sandbox)
            state = init_episode_state(task, mode, assignments, scout_obs)
        except ProtocolViolationError as err:
            base = EpisodeState(task_id=task.task_id, mode=mode, step=0)
            state = terminate_protocol_error(base, err, step=1)

        while not state.terminated:
            runnable = tuple(w for w in state.workers.values() if w.status == "running")
            if not runnable:
                state = replace(state, terminated=True, termination_reason="all_workers_done")
                break
            if state.step >= task.max_steps:
                state = replace(state, terminated=True, termination_reason="max_steps_reached")
                break

            # Advance active workers concurrently in deterministic round-robin order per tick.
            for worker in runnable:
                if state.terminated or state.step >= task.max_steps:
                    break
                next_step = state.step + 1
                try:
                    action = policy.next_worker_action(task, state, worker)
                except ProtocolViolationError as err:
                    state = terminate_protocol_error(state, err, step=next_step)
                    break

                state, issued_challenge = execute_worker_turn_step(
                    state,
                    task,
                    sandbox,
                    worker,
                    action,
                    next_step,
                    enforce_write_lock=enforce_write_lock,
                )
                if issued_challenge is not None:
                    if state.terminated:
                        break
                    try:
                        lead_resp = policy.respond_to_challenge(task, state, issued_challenge)
                    except ProtocolViolationError as err:
                        state = terminate_protocol_error(state, err, step=state.step + 1)
                        break
                    state = apply_lead_response(
                        state, lead_resp, issued_challenge.worker_id, state.step + 1
                    )
                    # Preemption pauses the old parallel batch; restart tick with replanned workers.
                    break

        public_final = sandbox.run_public_tests(state.step + 1, "harness")
        state = replace(state, public_tests_passed=(public_final.exit_code == 0))
        snapshot = dict(sandbox.snapshot_all())
        return state, snapshot
    finally:
        sandbox.close()
