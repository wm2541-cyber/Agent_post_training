"""Deterministic offline tests for the turn-based Interactive Arena & persistent JSON checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hidden_premise_lab.arena import (
    advance_arena,
    create_arena_session,
    load_arena,
    next_arena_actor,
    render_turn_prompt,
    save_arena,
    step_human_turn,
    step_llm_turn,
)
from hidden_premise_lab.arena import main as arena_main
from hidden_premise_lab.live_llm import LiveLLMPolicyTeam
from hidden_premise_lab.models import ProtocolViolationError
from hidden_premise_lab.orchestrator import run_episode
from hidden_premise_lab.tasks import build_legacy_token_task


def test_arena_checkpoint_roundtrip_hides_hidden_tests_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    session = create_arena_session(
        "legacy_token_v1",
        "bidirectional_escalation",
        human_actors=("lead",),
        model="deepseek-flash",
        max_steps=30,
    )
    ckpt = tmp_path / "match.json"
    save_arena(ckpt, session)

    raw_text = ckpt.read_text(encoding="utf-8")
    assert "hidden_tests/" not in raw_text
    assert "test_unique_nonces_and_tamper_resistance" not in raw_text

    loaded = load_arena(ckpt)
    assert loaded == session
    assert next_arena_actor(loaded) == "lead"

    corrupted = json.loads(raw_text)
    corrupted["state"]["lead_decisions"] = [[14, "override"]]  # Missing reason -> must raise!
    bad_path = tmp_path / "corrupted.json"
    bad_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="3-element"):
        load_arena(bad_path)

    bad_ver = json.loads(raw_text)
    bad_ver["schema_version"] = 99
    bad_ver_path = tmp_path / "bad_ver.json"
    bad_ver_path.write_text(json.dumps(bad_ver), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_arena(bad_ver_path)


def test_arena_human_lead_vs_mock_llm_workers_valid_whistle_replan(tmp_path: Path) -> None:
    task = build_legacy_token_task()
    passing_auth = task.initial_files["auth.py"].replace("DEFAULT_TTL = 300", "DEFAULT_TTL = 600")
    passing_client = (
        "from __future__ import annotations\n"
        "import uuid\nimport auth\n\n"
        "def build_auth_header(secret: str, user: str, now: int | None = None) -> dict:\n"
        "    ts = 1000 if now is None else now\n"
        "    return auth.sign_v2(secret, user, uuid.uuid4().hex, ts)\n"
    )

    def worker_only_chat(
        messages: tuple[dict[str, str], ...], model: str, base_url: str, api_key: str
    ) -> str:
        del model, base_url, api_key
        user_msg = messages[-1]["content"]
        if "Worker ID: w1" in user_msg:
            if "WROTE auth.py" not in user_msg:
                return json.dumps({"action": "write_file", "path": "auth.py", "content": passing_auth})
            return json.dumps({"action": "complete", "summary": "Updated DEFAULT_TTL to 600."})
        # w2 flow: read security policy on gen 0 -> whistle-blow -> on gen 1 write sign_v2 + complete
        if "Generation: 0" in user_msg:
            if "[step=" not in user_msg or "config/security_policy.json" not in user_msg:
                return json.dumps({"action": "read_file", "path": "config/security_policy.json"})
            return json.dumps(
                {
                    "action": "challenge",
                    "blocker_type": "PREMISE_INVALIDATED",
                    "invalidated_premise": "sign_v1 is disabled by SEC-002",
                    "evidence_file": "config/security_policy.json",
                    "evidence_excerpt": '"allow_v1_fallback": false',
                    "proposed_hint": "Switch client.py to auth.sign_v2",
                }
            )
        if "WROTE client.py" not in user_msg:
            return json.dumps({"action": "write_file", "path": "client.py", "content": passing_client})
        return json.dumps({"action": "complete", "summary": "Migrated client.py to auth.sign_v2."})

    log_file = tmp_path / "arena.jsonl"
    team = LiveLLMPolicyTeam(
        model="mock-flash",
        api_key="super-secret-arena-key",
        log_path=log_file,
        chat_fn=worker_only_chat,
    )
    ckpt = tmp_path / "lead_match.json"
    s0 = create_arena_session(
        "legacy_token_v1",
        "bidirectional_escalation",
        human_actors=("lead",),
        log_path=str(log_file),
        max_steps=30,
    )
    trap_dispatch = json.dumps(
        {
            "assignments": [
                {
                    "worker_id": "w1",
                    "goal": "Set DEFAULT_TTL = 600 in auth.py",
                    "explicit_premises": ["README v1 active"],
                    "writable_paths": ["auth.py"],
                },
                {
                    "worker_id": "w2",
                    "goal": "Implement build_auth_header using sign_v1",
                    "explicit_premises": ["README says use sign_v1"],
                    "writable_paths": ["client.py"],
                },
            ]
        }
    )
    s1 = step_human_turn(s0, trap_dispatch)
    assert s1.phase == "awaiting_worker"
    save_arena(ckpt, s1)

    s2, trace1 = advance_arena(load_arena(ckpt), team=team)
    assert s2.phase == "awaiting_lead_response"
    assert next_arena_actor(s2) == "lead"
    assert s2.pending_challenge is not None
    assert s2.pending_challenge.worker_id == "w2"
    assert len(trace1) == 4

    lead_prompt = render_turn_prompt(s2)
    assert lead_prompt is not None
    assert '"allow_v1_fallback": false' in str(lead_prompt["user"])

    replan_resp = json.dumps(
        {
            "decision": "replan",
            "reason": "SEC-002 forbids sign_v1; switch w2 to sign_v2.",
            "compressed_summary": "w1 already updated auth.py TTL; w2 must use sign_v2.",
            "assignments": [
                {
                    "worker_id": "w2",
                    "goal": "Implement client.py::build_auth_header using auth.sign_v2 and random nonce",
                    "explicit_premises": ["SEC-002 requires v2 tokens with unique nonces"],
                    "writable_paths": ["client.py"],
                }
            ],
        }
    )
    s3 = step_human_turn(s2, replan_resp)
    save_arena(ckpt, s3)
    s4, _ = advance_arena(load_arena(ckpt), team=team)

    assert s4.phase == "finished"
    assert s4.verdict is not None and s4.rewards is not None
    assert s4.verdict.public_passed is True
    assert s4.verdict.hidden_passed is True
    assert s4.rewards.outcome_label == "valid_whistle_replan"
    assert s4.rewards.per_worker_rewards["w2"] > 1.0
    assert s4.rewards.lead_reward > 0.8
    assert "super-secret-arena-key" not in ckpt.read_text(encoding="utf-8")
    assert "super-secret-arena-key" not in log_file.read_text(encoding="utf-8")


def test_arena_human_worker_vs_mock_llm_lead_write_locks_and_override(tmp_path: Path) -> None:
    def lead_and_w2_chat(
        messages: tuple[dict[str, str], ...], model: str, base_url: str, api_key: str
    ) -> str:
        del model, base_url, api_key
        sys_msg = messages[0]["content"]
        user_msg = messages[-1]["content"]
        if "Initial scout observations" in user_msg:
            return json.dumps(
                {
                    "assignments": [
                        {
                            "worker_id": "w1",
                            "goal": "Set DEFAULT_TTL = 600 in auth.py",
                            "explicit_premises": ["v1 active"],
                            "writable_paths": ["auth.py"],
                        },
                        {
                            "worker_id": "w2",
                            "goal": "Implement client.py",
                            "explicit_premises": ["v1 active"],
                            "writable_paths": ["client.py"],
                        },
                    ]
                }
            )
        if "reviewing an upward challenge" in sys_msg:
            return json.dumps(
                {
                    "decision": "override",
                    "reason": "Citation in client.py is already assigned to w2; finish auth.py.",
                }
            )
        return json.dumps({"action": "complete", "summary": "w2 completed stub."})

    team = LiveLLMPolicyTeam(model="mock-flash", api_key="k", chat_fn=lead_and_w2_chat)
    s0 = create_arena_session(
        "legacy_token_v1",
        "bidirectional_escalation",
        human_actors=("w1",),
        max_steps=30,
    )
    # LLM Lead dispatches w1 and w2, then stops right on w1's human turn.
    s1, _ = advance_arena(s0, team=team)
    assert s1.phase == "awaiting_worker"
    assert next_arena_actor(s1) == "w1"

    # Attempt cross-scope write outside w1's writable_paths -> mechanically refused!
    s2 = step_human_turn(
        s1,
        json.dumps({"action": "write_file", "path": "client.py", "content": "# hijacked"}),
    )
    assert "PERMISSION_DENIED" in s2.state.observations[-1].excerpt
    assert s2.workspace_files["client.py"] == s1.workspace_files["client.py"]

    # Advance lets LLM w2 act once, then stops back at w1!
    s3, _ = advance_arena(s2, team=team)
    assert next_arena_actor(s3) == "w1"

    # Attempt protected public_tests write -> mechanically refused!
    s4 = step_human_turn(
        s3,
        json.dumps({"action": "write_file", "path": "public_tests/test_public.py", "content": "# hack"}),
    )
    assert "protected test path" in s4.state.observations[-1].excerpt

    # Fake/weak challenge by w1 gets overridden by LLM Lead, delivering reason into local_notes.
    s5 = step_human_turn(s4, json.dumps({"action": "read_file", "path": "client.py"}))
    s6 = step_human_turn(
        s5,
        json.dumps(
            {
                "action": "challenge",
                "blocker_type": "MISSING_CONTRACT",
                "invalidated_premise": "client missing",
                "evidence_file": "client.py",
                "evidence_excerpt": "Implement client auth header",
            }
        ),
    )
    assert s6.phase == "awaiting_lead_response"
    s7, _ = advance_arena(s6, team=team)
    assert s7.phase == "awaiting_worker"
    assert next_arena_actor(s7) == "w1"
    assert any("already assigned to w2" in n for n in s7.state.workers["w1"].local_notes)


def test_arena_llm_protocol_error_and_cli_atomic_human_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. LLM protocol error terminates cleanly as protocol_error with heavy penalty.
    bad_team = LiveLLMPolicyTeam(
        model="mock-flash",
        api_key="k",
        max_protocol_retries=1,
        chat_fn=lambda *_: "{not valid json",
    )
    s0 = create_arena_session("solvable_no_conflict_trap", "fork_join")
    s1 = step_llm_turn(s0, team=bad_team)
    assert s1.phase == "finished"
    assert s1.state.termination_reason == "protocol_error"
    assert s1.verdict is not None and s1.rewards is not None
    assert s1.verdict.attribution_outcome == "protocol_error"
    assert s1.rewards.lead_reward < -0.9

    # 1b. Pure step_human_turn raises ProtocolViolationError on invalid JSON before mutation.
    s_human = create_arena_session("solvable_no_conflict_trap", "fork_join", human_actors=("lead",))
    with pytest.raises(ProtocolViolationError):
        step_human_turn(s_human, "{bad json")

    # 2. CLI human invalid JSON leaves saved JSON checkpoint untouched and returns exit code 2.
    ckpt = tmp_path / "cli_match.json"
    rc_tasks = arena_main(["tasks", "--json"])
    assert rc_tasks == 0
    out_tasks = json.loads(capsys.readouterr().out)
    assert [t["task_id"] for t in out_tasks] == [
        "legacy_token_v1",
        "invariant_cache_lock",
        "solvable_no_conflict_trap",
    ]

    rc_init = arena_main(
        [
            "init",
            "--task",
            "solvable_no_conflict_trap",
            "--mode",
            "fork_join",
            "--state",
            str(ckpt),
            "--human",
            "lead",
            "--human",
            "w1",
            "--json",
        ]
    )
    assert rc_init == 0
    _ = capsys.readouterr()
    before_bytes = ckpt.read_text(encoding="utf-8")

    rc_bad = arena_main(["step", "--state", str(ckpt), "--action-json", "{invalid"])
    assert rc_bad == 2
    assert "PROTOCOL_ERROR" in capsys.readouterr().err
    assert ckpt.read_text(encoding="utf-8") == before_bytes

    # Valid CLI steps finish the episode cleanly.
    dispatch_json = json.dumps(
        {
            "assignments": [
                {
                    "worker_id": "w1",
                    "goal": "Fix 1-based pagination",
                    "explicit_premises": ["1-based page"],
                    "writable_paths": ["paginator.py"],
                }
            ]
        }
    )
    assert arena_main(["step", "--state", str(ckpt), "--action-json", dispatch_json]) == 0
    fixed_paginator = (
        "from __future__ import annotations\n\n"
        "def paginate(items: list, page: int, page_size: int) -> list:\n"
        "    if page < 1 or page_size < 1:\n"
        "        raise ValueError('bad args')\n"
        "    start = (page - 1) * page_size\n"
        "    return items[start : start + page_size]\n"
    )
    assert (
        arena_main(
            [
                "step",
                "--state",
                str(ckpt),
                "--action-json",
                json.dumps({"action": "write_file", "path": "paginator.py", "content": fixed_paginator}),
            ]
        )
        == 0
    )
    assert (
        arena_main(
            [
                "step",
                "--state",
                str(ckpt),
                "--action-json",
                json.dumps({"action": "complete", "summary": "Fixed paginator."}),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert arena_main(["status", "--state", str(ckpt), "--json"]) == 0
    final_payload = json.loads(capsys.readouterr().out)
    assert final_payload["summary"]["phase"] == "finished"
    assert final_payload["summary"]["verdict"]["hidden_passed"] is True
    assert final_payload["summary"]["rewards"]["lead_reward"] > 0.7


def test_arena_parity_with_run_episode_deterministic_sequence() -> None:
    task = build_legacy_token_task()
    passing_auth = task.initial_files["auth.py"].replace("DEFAULT_TTL = 300", "DEFAULT_TTL = 600")
    replies = iter(
        [
            json.dumps(
                {
                    "assignments": [
                        {
                            "worker_id": "w1",
                            "goal": "Set DEFAULT_TTL = 600",
                            "explicit_premises": ["README v1"],
                            "writable_paths": ["auth.py"],
                        }
                    ]
                }
            ),
            json.dumps({"action": "write_file", "path": "auth.py", "content": passing_auth}),
            json.dumps({"action": "complete", "summary": "Updated auth.py"}),
        ]
    )
    replies_list = list(replies)

    idx_run = iter(replies_list)
    team_run = LiveLLMPolicyTeam(model="mock", api_key="k", chat_fn=lambda *_: next(idx_run))
    state_run, snap_run = run_episode(task, "fork_join", team_run)

    idx_arena = iter(replies_list)
    team_arena = LiveLLMPolicyTeam(model="mock", api_key="k", chat_fn=lambda *_: next(idx_arena))
    s0 = create_arena_session(task.task_id, "fork_join")
    s_end, _ = advance_arena(s0, team=team_arena)

    assert s_end.phase == "finished"
    assert s_end.state.step == state_run.step
    assert s_end.state.termination_reason == state_run.termination_reason
    assert s_end.state.public_tests_passed == state_run.public_tests_passed
    assert dict(s_end.workspace_files) == snap_run
