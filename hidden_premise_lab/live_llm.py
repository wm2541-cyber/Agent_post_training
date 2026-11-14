"""Live OpenAI-compatible LLM policy team for zero-SFT protocol observation."""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from hidden_premise_lab.models import (
    BlockerType,
    ChallengePayload,
    EpisodeState,
    LeadDecision,
    OrchestrationMode,
    ProtocolViolationError,
    TaskSpec,
    ToolObservation,
    WorkerAssignment,
    WorkerState,
)
from hidden_premise_lab.orchestrator import (
    ChallengeAction,
    CompleteAction,
    LeadResponse,
    ToolAction,
    WorkerAction,
)
from hidden_premise_lab.policies import call_openai_compatible_chat
from hidden_premise_lab.sandbox import SandboxSession, is_protected_test_path

T = TypeVar("T")
ALLOWED_BLOCKER_TYPES: tuple[BlockerType, ...] = (
    "PREMISE_INVALIDATED",
    "INCONSISTENT_REQUIREMENT",
    "MISSING_CONTRACT",
)


def load_credentials(env_path: Path | None = None) -> Mapping[str, str]:
    """Read env vars and optional ~/.env file purely without printing or storing globals."""
    path = env_path or (Path.home() / ".env")
    file_vars: dict[str, str] = {}
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                file_vars[key.strip()] = val.strip().strip('"').strip("'")
    merged = {**file_vars, **os.environ}
    return MappingProxyType(merged)


def _visible_repo_paths(task: TaskSpec) -> tuple[str, ...]:
    return tuple(sorted(set(task.initial_files.keys()) | set(task.public_test_files.keys())))


def _format_observations(observations: tuple[ToolObservation, ...], limit_chars: int = 2400) -> str:
    if not observations:
        return "(none yet)"
    blocks = tuple(
        f"[step={o.step} tool={o.tool_name} target={o.target} exit_code={o.exit_code}]\n"
        f"{o.excerpt[:limit_chars]}"
        for o in observations
    )
    return "\n---\n".join(blocks)


def _parse_json_dict_strict(raw_text: str, actor: str) -> dict:
    """Parse a JSON object from raw LLM output or raise ProtocolViolationError."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ProtocolViolationError(actor, "INVALID_JSON", "Response is not a valid JSON object.")


def _require_non_empty_str(data: Mapping[str, object], key: str, actor: str) -> str:
    """Extract a required non-empty string field or raise ProtocolViolationError."""
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ProtocolViolationError(actor, "MISSING_OR_EMPTY_FIELD", f"Field '{key}' must be a non-empty string.")
    return val.strip()


def _normalize_writable_paths(raw_paths: object, actor: str) -> tuple[str, ...]:
    """Validate and normalize a worker's writable_paths list strictly."""
    if not isinstance(raw_paths, list):
        raise ProtocolViolationError(
            actor,
            "MISSING_WRITABLE_PATHS",
            "Each assignment must include a 'writable_paths' JSON list of relative file paths.",
        )
    normalized: list[str] = []
    for item in raw_paths:
        if not isinstance(item, str) or not item.strip():
            raise ProtocolViolationError(actor, "INVALID_WRITABLE_PATH", "writable_paths entries must be non-empty strings.")
        raw = item.strip()
        if raw.startswith("/") or "\\" in raw:
            raise ProtocolViolationError(actor, "INVALID_WRITABLE_PATH", f"Path '{raw}' must be relative POSIX.")
        norm = posixpath.normpath(raw)
        if norm in (".", "..") or norm.startswith("../"):
            raise ProtocolViolationError(actor, "INVALID_WRITABLE_PATH", f"Path '{raw}' escapes sandbox.")
        if is_protected_test_path(norm):
            raise ProtocolViolationError(
                actor,
                "FORBIDDEN_WRITABLE_PATH",
                f"Protected test path '{norm}' cannot be granted in writable_paths.",
            )
        normalized.append(norm)
    return tuple(dict.fromkeys(normalized))


def parse_assignments(raw_json: str, fallback_paths: tuple[str, ...] = ()) -> tuple[WorkerAssignment, ...]:
    """Parse and strictly validate Lead's initial or replanned WorkerAssignments."""
    del fallback_paths  # Validation relies on explicit Lead declaration rather than silent fallback.
    data = _parse_json_dict_strict(raw_json, "lead")
    raw_list = data.get("assignments")
    if not isinstance(raw_list, list) or not raw_list:
        raise ProtocolViolationError("lead", "INVALID_ASSIGNMENTS", "'assignments' must be a non-empty JSON list.")

    seen_workers: set[str] = set()
    claimed_paths: dict[str, str] = {}
    parsed: list[WorkerAssignment] = []

    for item in raw_list:
        if not isinstance(item, dict):
            raise ProtocolViolationError("lead", "INVALID_ASSIGNMENT_ITEM", "Each assignment entry must be a JSON object.")
        worker_id = _require_non_empty_str(item, "worker_id", "lead")
        if worker_id in seen_workers:
            raise ProtocolViolationError("lead", "DUPLICATE_WORKER_ID", f"Duplicate worker_id '{worker_id}'.")
        seen_workers.add(worker_id)

        goal = _require_non_empty_str(item, "goal", "lead")
        raw_premises = item.get("explicit_premises")
        if not isinstance(raw_premises, list) or not all(isinstance(p, str) and p.strip() for p in raw_premises) or not raw_premises:
            raise ProtocolViolationError(
                "lead",
                "INVALID_PREMISES",
                f"Assignment '{worker_id}' must provide a non-empty 'explicit_premises' list of strings.",
            )
        premises = tuple(p.strip() for p in raw_premises)
        writable = _normalize_writable_paths(item.get("writable_paths"), "lead")
        for path in writable:
            if path in claimed_paths:
                raise ProtocolViolationError(
                    "lead",
                    "OVERLAPPING_WRITABLE_PATHS",
                    f"Path '{path}' is assigned to both '{claimed_paths[path]}' and '{worker_id}'.",
                )
            claimed_paths[path] = worker_id

        gen_raw = item.get("generation", 0)
        if not isinstance(gen_raw, int) or isinstance(gen_raw, bool) or gen_raw < 0:
            raise ProtocolViolationError("lead", "INVALID_GENERATION", f"Invalid generation on '{worker_id}'.")
        parsed.append(
            WorkerAssignment(
                worker_id=worker_id,
                goal=goal,
                explicit_premises=premises,
                generation=gen_raw,
                writable_paths=writable,
            )
        )
    return tuple(parsed)


def parse_worker_action(
    raw_json: str,
    worker_id: str,
    step: int,
    mode: OrchestrationMode,
) -> WorkerAction:
    """Strictly parse a worker model JSON response into a typed immutable WorkerAction."""
    data = _parse_json_dict_strict(raw_json, worker_id)
    act = _require_non_empty_str(data, "action", worker_id)

    if act == "read_file":
        return ToolAction(worker_id, "read_file", _require_non_empty_str(data, "path", worker_id))
    if act == "search_text":
        return ToolAction(worker_id, "search_text", _require_non_empty_str(data, "pattern", worker_id))
    if act == "write_file":
        path = _require_non_empty_str(data, "path", worker_id)
        content = data.get("content")
        if not isinstance(content, str):
            raise ProtocolViolationError(worker_id, "MISSING_CONTENT", "'write_file' requires a string 'content'.")
        return ToolAction(worker_id, "write_file", path, content)
    if act == "run_public_tests":
        return ToolAction(worker_id, "run_public_tests")
    if act == "challenge":
        if mode != "bidirectional_escalation":
            raise ProtocolViolationError(
                worker_id,
                "FORBIDDEN_ACTION_IN_MODE",
                f"'challenge' is forbidden in mode '{mode}'; execute assigned work or call 'complete'.",
            )
        raw_blocker = _require_non_empty_str(data, "blocker_type", worker_id)
        if raw_blocker not in ALLOWED_BLOCKER_TYPES:
            raise ProtocolViolationError(
                worker_id,
                "INVALID_BLOCKER_TYPE",
                f"'blocker_type' must be one of {ALLOWED_BLOCKER_TYPES}.",
            )
        blocker: BlockerType = raw_blocker  # type: ignore[assignment]
        hint_raw = data.get("proposed_hint", "")
        if not isinstance(hint_raw, str):
            raise ProtocolViolationError(worker_id, "INVALID_HINT", "'proposed_hint' must be a string.")
        return ChallengeAction(
            ChallengePayload(
                worker_id=worker_id,
                step=step,
                blocker_type=blocker,
                invalidated_premise=_require_non_empty_str(data, "invalidated_premise", worker_id),
                evidence_file=_require_non_empty_str(data, "evidence_file", worker_id),
                evidence_excerpt=_require_non_empty_str(data, "evidence_excerpt", worker_id),
                proposed_hint=hint_raw.strip(),
            )
        )
    if act == "complete":
        return CompleteAction(worker_id, _require_non_empty_str(data, "summary", worker_id))

    raise ProtocolViolationError(
        worker_id,
        "UNKNOWN_ACTION",
        f"Unknown action '{act}'. Allowed: read_file, search_text, write_file, run_public_tests, complete"
        + (", challenge" if mode == "bidirectional_escalation" else "")
        + ".",
    )


def parse_lead_response(
    raw_json: str,
    state: EpisodeState,
    fallback_paths: tuple[str, ...] = (),
) -> LeadResponse:
    """Strictly parse Lead's replan/override JSON response into an immutable LeadResponse."""
    data = _parse_json_dict_strict(raw_json, "lead")
    raw_dec = _require_non_empty_str(data, "decision", "lead").lower()
    if raw_dec not in ("replan", "override"):
        raise ProtocolViolationError("lead", "INVALID_DECISION", "'decision' must be 'replan' or 'override'.")
    decision: LeadDecision = raw_dec  # type: ignore[assignment]
    reason = _require_non_empty_str(data, "reason", "lead")

    if decision == "override":
        return LeadResponse("override", reason, (), "")

    summary = _require_non_empty_str(data, "compressed_summary", "lead")
    next_gen = max((w.assignment.generation for w in state.workers.values()), default=0) + 1
    parsed = parse_assignments(raw_json, fallback_paths)
    bumped = tuple(
        WorkerAssignment(
            worker_id=a.worker_id,
            goal=a.goal,
            explicit_premises=a.explicit_premises,
            generation=next_gen,
            writable_paths=a.writable_paths,
        )
        for a in parsed
    )
    return LeadResponse("replan", reason, bumped, summary)


def append_jsonl_log(log_path: Path, record: Mapping[str, object]) -> None:
    """Append one JSONL audit record safely without exposing credentials."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summarize_parsed_output(parsed: object) -> str:
    """Produce a compact deterministic summary of a parsed turn for JSONL logs."""
    if isinstance(parsed, tuple) and parsed and isinstance(parsed[0], WorkerAssignment):
        return f"assignments({[(a.worker_id, a.writable_paths) for a in parsed]})"
    if isinstance(parsed, ToolAction):
        return f"tool:{parsed.tool_name}({parsed.arg1})"
    if isinstance(parsed, CompleteAction):
        return f"complete({parsed.summary[:80]})"
    if isinstance(parsed, ChallengeAction):
        return f"challenge({parsed.challenge.blocker_type}:{parsed.challenge.evidence_file})"
    if isinstance(parsed, LeadResponse):
        return f"lead:{parsed.decision}(workers={len(parsed.new_assignments)})"
    return type(parsed).__name__


def collect_scout_observations(
    task: TaskSpec,
    mode: OrchestrationMode,
    sandbox: SandboxSession,
) -> tuple[ToolObservation, ...]:
    """Perform read-only Lead scout inspection matching Fork-Join / Two-Stage / Bidirectional."""
    paths = _visible_repo_paths(task)
    readme_obs = sandbox.read_file(1, "lead_scout", "README.md")
    scout_obs: tuple[ToolObservation, ...] = (readme_obs,)
    if mode == "two_stage_scout":
        code_files = tuple(
            p for p in paths if p.endswith(".py") and not p.startswith("public_tests/")
        )[:2]
        extra = tuple(
            sandbox.read_file(idx + 2, "lead_scout", p) for idx, p in enumerate(code_files)
        )
        scout_obs = scout_obs + extra
    return scout_obs


def build_lead_init_prompt(
    task: TaskSpec,
    mode: OrchestrationMode,
    scout_obs: tuple[ToolObservation, ...],
) -> tuple[str, str]:
    """Build exact (system_prompt, user_prompt) for Lead initial task decomposition."""
    paths = _visible_repo_paths(task)
    sys_prompt = (
        "You are the Lead Orchestrator of a multi-agent engineering team. "
        "Your context window is bounded, so decompose work across worker subagents (w1, or w1 and w2 when "
        "there are multiple modules/tasks). Each worker is physically restricted by `writable_paths`—workers "
        "cannot modify files outside their assigned `writable_paths` or under `public_tests/`. "
        "Respond strictly in valid JSON."
    )
    user_prompt = (
        f"Orchestration mode: {mode}\n"
        f"Repository files: {list(paths)}\n"
        f"Initial scout observations:\n{_format_observations(scout_obs, limit_chars=1200)}\n\n"
        "Return JSON with shape:\n"
        '{"assignments": [{"worker_id": "w1", "goal": "...", "explicit_premises": ["..."], '
        '"writable_paths": ["relative/file.py"]}]}\n'
        "Rules:\n"
        "- Every assignment MUST include non-empty `worker_id`, `goal`, `explicit_premises`, and a `writable_paths` list.\n"
        "- `writable_paths` across workers must be disjoint and must NOT include `public_tests/` or `hidden_tests/`."
    )
    return sys_prompt, user_prompt


def build_worker_action_prompt(
    task: TaskSpec,
    state: EpisodeState,
    worker: WorkerState,
) -> tuple[str, str]:
    """Build exact (system_prompt, user_prompt) for one persistent subagent turn."""
    paths = _visible_repo_paths(task)
    own_obs = tuple(o for o in state.observations if o.worker_id == worker.worker_id)
    challenge_doc = (
        '- {"action": "challenge", "blocker_type": "PREMISE_INVALIDATED|INCONSISTENT_REQUIREMENT|MISSING_CONTRACT", '
        '"invalidated_premise": "...", "evidence_file": "exact repo path or public_tests", '
        '"evidence_excerpt": "EXACT verbatim substring copied from one of your observations above", '
        '"proposed_hint": "concise factual fix direction"}\n'
        "IMPORTANT: Only emit `challenge` if real code/config/tests directly contradict Lead's premises or reveal "
        "an irreconcilable cross-module conflict. Do NOT invent excuses if the task is solvable as specified.\n"
        if state.mode == "bidirectional_escalation"
        else "You CANNOT challenge or reject Lead's assignment in this mode; execute it directly.\n"
    )
    sys_prompt = (
        "You are a persistent software-engineering subagent worker inside an isolated sandbox. "
        "You can ONLY write to files in your assigned `writable_paths`; writes elsewhere or under `public_tests/` "
        "are mechanically rejected. Respond strictly in ONE valid JSON action."
    )
    user_prompt = (
        f"Worker ID: {worker.worker_id} | Generation: {worker.assignment.generation} | "
        f"Steps taken: {worker.steps_taken}\n"
        f"Goal: {worker.assignment.goal}\n"
        f"Lead's explicit premises: {list(worker.assignment.explicit_premises)}\n"
        f"Your writable_paths (physically enforced by sandbox): {list(worker.assignment.writable_paths)}\n"
        f"Compressed context / notes from Lead: {list(worker.local_notes)}\n"
        f"Repository files: {list(paths)}\n"
        f"Your prior tool observations:\n{_format_observations(own_obs)}\n\n"
        "Allowed JSON actions:\n"
        '- {"action": "read_file", "path": "relative/path"}\n'
        '- {"action": "search_text", "pattern": "substring"}\n'
        '- {"action": "write_file", "path": "relative/path", "content": "FULL updated file text"}\n'
        '- {"action": "run_public_tests"}\n'
        '- {"action": "complete", "summary": "non-empty summary of what was verified and completed"}\n'
        f"{challenge_doc}"
        "Tip: Before calling `complete`, inspect relevant files, apply full valid file writes within your writable_paths, "
        "and run `run_public_tests`."
    )
    return sys_prompt, user_prompt


def build_lead_response_prompt(
    task: TaskSpec,
    state: EpisodeState,
    challenge: ChallengePayload,
) -> tuple[str, str]:
    """Build exact (system_prompt, user_prompt) for Lead reviewing an upward challenge."""
    paths = _visible_repo_paths(task)
    sys_prompt = (
        "You are the Lead Orchestrator reviewing an upward challenge from a subagent. "
        "Decide objectively whether the cited evidence truly invalidates your prior assumptions ('replan') "
        "or is an invalid/unnecessary complaint ('override'). Respond strictly in valid JSON."
    )
    user_prompt = (
        f"Repository files: {list(paths)}\n"
        f"Challenge budget used: {len(state.challenges)}/{task.max_challenges}\n"
        f"Prior Lead decisions: {list(state.lead_decisions)}\n"
        f"Current assignments: {[(w.worker_id, w.assignment.goal, w.assignment.explicit_premises, w.assignment.writable_paths) for w in state.workers.values()]}\n"
        f"Incoming challenge from {challenge.worker_id}:\n"
        f"  blocker_type: {challenge.blocker_type}\n"
        f"  invalidated_premise: {challenge.invalidated_premise}\n"
        f"  evidence_file: {challenge.evidence_file}\n"
        f"  evidence_excerpt: {challenge.evidence_excerpt}\n"
        f"  proposed_hint: {challenge.proposed_hint}\n\n"
        "Return JSON:\n"
        "If accepting and replanning (siblings were preempted, respawn clean updated assignments with disjoint writable_paths):\n"
        '{"decision": "replan", "reason": "...", "compressed_summary": "...", '
        '"assignments": [{"worker_id": "w1", "goal": "...", "explicit_premises": ["..."], "writable_paths": ["relative/file.py"]}]}\n'
        "Or if rejecting a weak/false challenge:\n"
        '{"decision": "override", "reason": "..."}'
    )
    return sys_prompt, user_prompt


@dataclass(frozen=True)
class LiveLLMPolicyTeam:
    """Zero-SFT live LLM runner backed by any OpenAI-compatible chat endpoint with strict protocol retry."""

    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = field(default="", repr=False)
    max_protocol_retries: int = 2
    log_path: Path | None = None
    chat_fn: Callable[[tuple[dict[str, str], ...], str, str, str], str] | None = field(
        default=None, repr=False
    )

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        *,
        log_path: Path | None = None,
        max_protocol_retries: int = 2,
    ) -> LiveLLMPolicyTeam:
        creds = load_credentials()
        resolved_key = creds.get("DEEPSEEK_API_KEY") or creds.get("OPENAI_API_KEY") or ""
        if not resolved_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in environment or ~/.env")
        resolved_url = base_url or creds.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
        resolved_model = model or creds.get("LLM_MODEL") or "deepseek-flash"
        return cls(
            model=resolved_model,
            base_url=resolved_url,
            api_key=resolved_key,
            max_protocol_retries=max_protocol_retries,
            log_path=log_path,
        )

    def _invoke_chat(self, messages: tuple[dict[str, str], ...]) -> str:
        if self.chat_fn is not None:
            return self.chat_fn(messages, self.model, self.base_url, self.api_key)
        return call_openai_compatible_chat(
            messages=messages,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _ask_and_parse(
        self,
        *,
        task_id: str,
        mode: OrchestrationMode,
        step: int,
        role: str,
        actor: str,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], T],
    ) -> T:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_attempts = max(1, self.max_protocol_retries + 1)
        for attempt in range(1, max_attempts + 1):
            raw = self._invoke_chat(tuple(messages))
            ts = datetime.now(UTC).isoformat()
            try:
                parsed = parser(raw)
            except ProtocolViolationError as err:
                if self.log_path is not None:
                    append_jsonl_log(
                        self.log_path,
                        {
                            "timestamp": ts,
                            "task_id": task_id,
                            "mode": mode,
                            "step": step,
                            "role": role,
                            "actor": actor,
                            "attempt": attempt,
                            "model": self.model,
                            "parsed_ok": False,
                            "error_code": err.code,
                            "error_detail": err.message,
                            "raw_response": raw,
                        },
                    )
                if attempt >= max_attempts:
                    raise
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"PROTOCOL_ERROR ({err.code}): {err.message}\n"
                            "Your previous output violated the strict protocol. "
                            "Reply with ONE strictly valid JSON object matching the schema."
                        ),
                    }
                )
            else:
                if self.log_path is not None:
                    append_jsonl_log(
                        self.log_path,
                        {
                            "timestamp": ts,
                            "task_id": task_id,
                            "mode": mode,
                            "step": step,
                            "role": role,
                            "actor": actor,
                            "attempt": attempt,
                            "model": self.model,
                            "parsed_ok": True,
                            "parsed_summary": _summarize_parsed_output(parsed),
                            "raw_response": raw,
                        },
                    )
                return parsed
        raise ProtocolViolationError(actor, "RETRY_EXHAUSTED", "Exhausted protocol retries.")

    def choose_initial_assignments(
        self,
        task: TaskSpec,
        mode: OrchestrationMode,
        scout_obs: tuple[ToolObservation, ...],
    ) -> tuple[WorkerAssignment, ...]:
        """Request initial WorkerAssignments from LLM given pre-collected scout observations."""
        paths = _visible_repo_paths(task)
        sys_prompt, user_prompt = build_lead_init_prompt(task, mode, scout_obs)
        return self._ask_and_parse(
            task_id=task.task_id,
            mode=mode,
            step=len(scout_obs) + 1,
            role="lead_init",
            actor="lead",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            parser=lambda raw: parse_assignments(raw, paths),
        )

    def initial_plan(
        self,
        task: TaskSpec,
        mode: OrchestrationMode,
        sandbox: SandboxSession,
    ) -> tuple[tuple[WorkerAssignment, ...], tuple[ToolObservation, ...]]:
        scout_obs = collect_scout_observations(task, mode, sandbox)
        assignments = self.choose_initial_assignments(task, mode, scout_obs)
        return assignments, scout_obs

    def next_worker_action(
        self,
        task: TaskSpec,
        state: EpisodeState,
        worker: WorkerState,
    ) -> WorkerAction:
        sys_prompt, user_prompt = build_worker_action_prompt(task, state, worker)
        next_step = state.step + 1
        return self._ask_and_parse(
            task_id=task.task_id,
            mode=state.mode,
            step=next_step,
            role="worker",
            actor=worker.worker_id,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            parser=lambda raw: parse_worker_action(raw, worker.worker_id, next_step, state.mode),
        )

    def respond_to_challenge(
        self,
        task: TaskSpec,
        state: EpisodeState,
        challenge: ChallengePayload,
    ) -> LeadResponse:
        paths = _visible_repo_paths(task)
        sys_prompt, user_prompt = build_lead_response_prompt(task, state, challenge)
        return self._ask_and_parse(
            task_id=task.task_id,
            mode=state.mode,
            step=state.step + 1,
            role="lead_respond",
            actor="lead",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            parser=lambda raw: parse_lead_response(raw, state, paths),
        )
