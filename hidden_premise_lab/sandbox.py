"""Isolated workspace materializer and tool-execution boundary for sandbox verification."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hidden_premise_lab.models import TaskSpec, ToolObservation, freeze_mapping


def sha256_text(text: str) -> str:
    """Compute a deterministic SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_public_test_hashes(task: TaskSpec) -> Mapping[str, str]:
    """Snapshot immutable hashes of public test files for tamper detection."""
    return freeze_mapping({path: sha256_text(content) for path, content in task.public_test_files.items()})


def is_protected_test_path(rel_posix: str) -> bool:
    """Return True if a normalized POSIX path resides under public or hidden test directories."""
    parts = rel_posix.split("/")
    return bool(parts and parts[0] in ("public_tests", "hidden_tests"))


def resolve_sandbox_rel_path(root: Path, rel_path: str) -> tuple[Path, str] | None:
    """Resolve a relative path safely inside root; return None on empty, absolute, or escaping paths."""
    cleaned = rel_path.strip()
    if not cleaned:
        return None
    raw = Path(cleaned)
    if raw.is_absolute():
        return None
    root_resolved = root.resolve()
    target = (root_resolved / raw).resolve()
    if not target.is_relative_to(root_resolved) or target == root_resolved:
        return None
    return target, target.relative_to(root_resolved).as_posix()


def materialize_files(root: Path, files: Mapping[str, str]) -> None:
    """Write a mapping of relative paths into an isolated directory."""
    for rel_path, content in files.items():
        resolved = resolve_sandbox_rel_path(root, rel_path)
        if resolved is None:
            raise ValueError(f"Unsafe path traversal blocked: {rel_path}")
        target, _ = resolved
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def snapshot_directory(root: Path, relative_paths: tuple[str, ...] | None = None) -> Mapping[str, str]:
    """Capture an immutable mapping of file contents from a sandbox directory."""
    if relative_paths is not None:
        entries = {}
        for rel in relative_paths:
            candidate = root / rel
            if candidate.is_file():
                entries[rel] = candidate.read_text(encoding="utf-8")
        return freeze_mapping(entries)
    collected = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    return freeze_mapping(collected)


@dataclass(frozen=True)
class SubprocessOutcome:
    """Captured stdout/stderr/exit status from running a verifier suite."""

    exit_code: int
    output: str


def run_unittest_suite(workspace: Path, test_dir: str) -> SubprocessOutcome:
    """Execute unittest discovery inside an isolated directory without shell injection."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", test_dir, "-v"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SubprocessOutcome(exit_code=124, output=f"TIMEOUT: unittest suite '{test_dir}' exceeded 15s")
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return SubprocessOutcome(exit_code=proc.returncode, output=combined)


def evaluate_in_isolated_tempdir(
    task: TaskSpec,
    final_workspace_files: Mapping[str, str],
) -> tuple[bool, bool, str, tuple[str, ...]]:
    """Run integrity checks, public tests, and withheld hidden tests in a clean temp sandbox."""
    expected_hashes = compute_public_test_hashes(task)
    modified_or_missing = tuple(
        path
        for path, expected in expected_hashes.items()
        if sha256_text(final_workspace_files.get(path, "")) != expected
    )
    extra_protected = tuple(
        sorted(
            path
            for path in final_workspace_files
            if is_protected_test_path(path.replace("\\", "/")) and path not in expected_hashes
        )
    )
    tampered = modified_or_missing + extra_protected
    with tempfile.TemporaryDirectory(prefix=f"eval_{task.task_id}_") as tmp_dir:
        eval_root = Path(tmp_dir)
        # Re-materialize agent code changes, excluding hidden paths first, then mount trusted tests.
        safe_agent_files = {
            k: v
            for k, v in final_workspace_files.items()
            if not k.startswith("hidden_tests/")
        }
        materialize_files(eval_root, safe_agent_files)
        public_res = run_unittest_suite(eval_root, "public_tests")

        # Mount withheld hidden_tests and restore canonical public_tests before final grading
        # so hidden evaluation never inherits test-tampered assertions.
        for extra in extra_protected:
            extra_target = eval_root / extra
            if extra_target.is_file():
                extra_target.unlink()
        materialize_files(eval_root, task.public_test_files)
        materialize_files(eval_root, task.hidden_test_files)
        hidden_res = run_unittest_suite(eval_root, "hidden_tests")

    return (
        public_res.exit_code == 0,
        hidden_res.exit_code == 0,
        hidden_res.output,
        tampered,
    )


class SandboxSession:
    """Imperative-shell wrapper around a single temporary directory during rollout."""

    def __init__(
        self,
        task: TaskSpec,
        initial_workspace: Mapping[str, str] | None = None,
    ) -> None:
        self._task = task
        self._tmp = tempfile.TemporaryDirectory(prefix=f"rollout_{task.task_id}_")
        self.root = Path(self._tmp.name)
        # Crucially withhold hidden_test_files from the agent sandbox (Kimi K3 AET isolation).
        if initial_workspace is not None:
            safe_files = {
                k: v for k, v in initial_workspace.items() if not k.startswith("hidden_tests/")
            }
            materialize_files(self.root, safe_files)
        else:
            materialize_files(self.root, task.initial_files)
            materialize_files(self.root, task.public_test_files)

    def close(self) -> None:
        self._tmp.cleanup()

    def _resolve_existing_or_target(self, rel_path: str) -> tuple[Path, str] | None:
        return resolve_sandbox_rel_path(self.root, rel_path)

    def read_file(self, step: int, worker_id: str, rel_path: str) -> ToolObservation:
        resolved = self._resolve_existing_or_target(rel_path)
        if resolved is None:
            return ToolObservation(
                step,
                worker_id,
                "read_file",
                rel_path,
                1,
                f"PERMISSION_DENIED: unsafe or invalid path '{rel_path}'",
                "",
            )
        target, norm_rel = resolved
        if not target.is_file():
            return ToolObservation(step, worker_id, "read_file", norm_rel, 1, f"FileNotFound: {norm_rel}", "")
        content = target.read_text(encoding="utf-8")
        return ToolObservation(step, worker_id, "read_file", norm_rel, 0, content, sha256_text(content))

    def search_text(self, step: int, worker_id: str, pattern: str) -> ToolObservation:
        hits: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(self.root).as_posix()
                for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    if pattern in line:
                        hits.append(f"{rel}:{idx}: {line}")
        excerpt = "\n".join(hits) if hits else f"NO_MATCH:{pattern}"
        return ToolObservation(
            step,
            worker_id,
            "search_text",
            pattern,
            0 if hits else 1,
            excerpt,
            sha256_text(excerpt),
        )

    def write_file(
        self,
        step: int,
        worker_id: str,
        rel_path: str,
        content: str,
        allowed_paths: tuple[str, ...] | None = None,
    ) -> ToolObservation:
        resolved = self._resolve_existing_or_target(rel_path)
        if resolved is None:
            return ToolObservation(
                step,
                worker_id,
                "write_file",
                rel_path,
                1,
                f"PERMISSION_DENIED: unsafe or invalid path '{rel_path}'",
                "",
            )
        target, norm_rel = resolved
        if allowed_paths is not None:
            if is_protected_test_path(norm_rel):
                return ToolObservation(
                    step,
                    worker_id,
                    "write_file",
                    norm_rel,
                    1,
                    f"PERMISSION_DENIED: '{norm_rel}' is a protected test path",
                    "",
                )
            norm_allowed = tuple(
                res[1]
                for p in allowed_paths
                if (res := resolve_sandbox_rel_path(self.root, p)) is not None
                and not is_protected_test_path(res[1])
            )
            if norm_rel not in norm_allowed:
                return ToolObservation(
                    step,
                    worker_id,
                    "write_file",
                    norm_rel,
                    1,
                    f"PERMISSION_DENIED: '{norm_rel}' is outside writable_paths={list(norm_allowed)}",
                    "",
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolObservation(step, worker_id, "write_file", norm_rel, 0, f"WROTE {norm_rel}", sha256_text(content))

    def run_public_tests(self, step: int, worker_id: str) -> ToolObservation:
        outcome = run_unittest_suite(self.root, "public_tests")
        return ToolObservation(
            step,
            worker_id,
            "run_public_tests",
            "public_tests",
            outcome.exit_code,
            outcome.output,
            sha256_text(outcome.output),
        )

    def snapshot_all(self) -> Mapping[str, str]:
        return snapshot_directory(self.root)
