"""CLI benchmark runner comparing orchestration paradigms and anti-reward-hacking defenses."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from hidden_premise_lab.judge import evaluate_episode
from hidden_premise_lab.live_llm import LiveLLMPolicyTeam
from hidden_premise_lab.models import OrchestrationMode, TaskSpec
from hidden_premise_lab.orchestrator import PolicyTeam, run_episode
from hidden_premise_lab.policies import ScriptedPolicyTeam
from hidden_premise_lab.reward import compute_rewards
from hidden_premise_lab.tasks import all_benchmark_tasks


def _eval_single_case(
    label: str,
    mode: OrchestrationMode,
    policy: PolicyTeam,
    task: TaskSpec,
    *,
    enforce_write_lock: bool = True,
) -> dict:
    """Run and evaluate one (policy, task) pair purely via isolated sandbox execution."""
    state, snapshot = run_episode(task, mode, policy, enforce_write_lock=enforce_write_lock)
    verdict = evaluate_episode(task, state, snapshot)
    rewards = compute_rewards(verdict, state)
    return {
        "strategy": label,
        "mode": mode,
        "task_id": task.task_id,
        "has_conflict": task.has_true_premise_conflict,
        "public_ok": verdict.public_passed,
        "hidden_ok": verdict.hidden_passed,
        "intact": verdict.sandbox_intact,
        "challenges": len(state.challenges),
        "crit_cost": verdict.critical_path_cost,
        "outcome": verdict.attribution_outcome,
        "r_lead": rewards.lead_reward,
        "r_sub": rewards.subagent_reward,
        "notes": rewards.notes,
        "trace": {
            "termination_reason": state.termination_reason,
            "events": [f"s{e.step}:{e.actor}:{e.action}({e.detail[:80]})" for e in state.events],
            "challenges": [
                {
                    "worker": c.worker_id,
                    "step": c.step,
                    "blocker": c.blocker_type,
                    "file": c.evidence_file,
                    "excerpt": c.evidence_excerpt[:120],
                }
                for c in state.challenges
            ],
            "audits": [
                {
                    "grounded": a.evidence_grounded,
                    "verified": a.conflict_verified,
                    "reason": a.reason,
                }
                for a in verdict.challenge_audits
            ],
            "lead_decisions": [list(d) for d in state.lead_decisions],
        },
    }


def run_benchmark_matrix(
    tasks: tuple[TaskSpec, ...] | None = None,
) -> tuple[dict, ...]:
    """Execute the full canonical matrix of baselines, bidirectional teams, and hacking tests."""
    suite = tasks or all_benchmark_tasks()
    configs: tuple[tuple[str, OrchestrationMode, PolicyTeam], ...] = (
        ("Fork-Join (PARL-like)", "fork_join", ScriptedPolicyTeam("fork_join_obedient")),
        ("Two-Stage Scout", "two_stage_scout", ScriptedPolicyTeam("two_stage_scout")),
        ("Bidirectional (Ours)", "bidirectional_escalation", ScriptedPolicyTeam("bidirectional_honest")),
        ("Override Lead (Failure)", "bidirectional_escalation", ScriptedPolicyTeam("stubborn_override_lead")),
        ("Slacker Fake-Challenge", "bidirectional_escalation", ScriptedPolicyTeam("defensive_slacker_hacking")),
        ("Test-Tamper Cheater", "fork_join", ScriptedPolicyTeam("test_tamper_cheater")),
    )
    return tuple(
        _eval_single_case(
            label,
            mode,
            policy,
            task,
            # Bypass Layer-0 write lock only for Test-Tamper Cheater so Layer-1 judge detection remains exercised.
            enforce_write_lock=(label != "Test-Tamper Cheater"),
        )
        for label, mode, policy in configs
        for task in suite
    )


def run_live_matrix(
    model: str | None = None,
    base_url: str | None = None,
    tasks: tuple[TaskSpec, ...] | None = None,
    log_path: Path | None = None,
) -> tuple[dict, ...]:
    """Run live zero-SFT LLM comparison across Fork-Join, Two-Stage Scout, and Bidirectional modes."""
    resolved_log = log_path or (
        Path(__file__).resolve().parent.parent
        / "logs"
        / f"live_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    print(f"[live raw log] {resolved_log}", file=sys.stderr)
    team = LiveLLMPolicyTeam.from_env(model=model, base_url=base_url, log_path=resolved_log)
    suite = tuple(replace(t, max_steps=30) for t in (tasks or all_benchmark_tasks()))
    modes: tuple[tuple[str, OrchestrationMode], ...] = (
        (f"Live {team.model} (Fork-Join)", "fork_join"),
        (f"Live {team.model} (Two-Stage Scout)", "two_stage_scout"),
        (f"Live {team.model} (Bidirectional)", "bidirectional_escalation"),
    )
    return tuple(
        _eval_single_case(label, mode, team, task)
        for label, mode in modes
        for task in suite
    )


def format_markdown_table(rows: tuple[dict, ...]) -> str:
    """Format benchmark matrix rows as a crisp Markdown comparison table."""
    header = (
        "| 策略 / 角色组合 | 任务 ID | 真实冲突 | Public测试 | Hidden终验 | 沙盒完整 | "
        "退回次数 | 关键路径成本 | 结束原因 | 归因判定 | R_lead | R_sub |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"
    )
    lines = [
        f"| {r['strategy']} | `{r['task_id']}` | {'是' if r['has_conflict'] else '否'} | "
        f"{'PASS' if r['public_ok'] else 'FAIL'} | {'PASS' if r['hidden_ok'] else 'FAIL'} | "
        f"{'OK' if r['intact'] else 'TAMPER'} | {r['challenges']} | {r['crit_cost']:.1f} | "
        f"`{r['trace']['termination_reason']}` | `{r['outcome']}` | {r['r_lead']:+.2f} | {r['r_sub']:+.2f} |"
        for r in rows
    ]
    return "\n".join([header, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description="Hidden Premise Conflict Benchmark & Judge Harness")
    parser.add_argument("--mode", choices=("simulate", "live"), default="simulate")
    parser.add_argument("--model", default=None, help="Live LLM model id (e.g. deepseek-flash)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--log-file", default=None, help="Path for raw LLM JSONL turn logs (live mode)")
    parser.add_argument("--json", action="store_true", help="Print JSON output instead of Markdown table")
    args = parser.parse_args()

    rows = (
        run_live_matrix(
            model=args.model,
            base_url=args.base_url,
            log_path=Path(args.log_file) if args.log_file else None,
        )
        if args.mode == "live"
        else run_benchmark_matrix()
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
