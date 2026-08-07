#!/usr/bin/env python3
"""Bounded OpenAI reasoning bridge for the DCR Codex collaboration.

Install the only optional runtime dependency with:

    python -m pip install openai

The OpenAI SDK reads OPENAI_API_KEY directly from the environment. This
program never reads the key from a file and never prints or persists it.

Version one performs one reasoning handoff per invocation. It does not call
Codex, edit solver source, run tests, run the solver, commit, push, or merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EXPECTED_BRANCH = "adaptive-recycling-closure"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_INPUT_BYTES = 120_000
DEFAULT_MAX_ARTIFACT_BYTES = 32_000
MAX_INPUT_BYTES_CAP = 500_000
MAX_ARTIFACT_BYTES_CAP = 64_000
MAX_CYCLES_CAP = 3

COLLABORATION_FILES = (
    ".dcr-agent/RULES.md",
    ".dcr-agent/STATUS.md",
    ".dcr-agent/HISTORY.md",
    ".dcr-agent/CHATGPT_NEXT.md",
)

PERMANENT_INSTRUCTION = """You are the scientific/numerical reasoning side of the DCR solver collaboration.

Primary goal:
Given prescribed u(x) and T(x), obtain a reliable, fully audited steady DCR solution.

Treat RULES.md as binding.

Codex has already executed the experiment described by STATUS.md. Analyze verified observations, not speculation.

Decide exactly ONE next discriminating experiment.

Do not change physical equations without explicit human approval.

Do not change accepted checkpoints.

Do not propose multiple experiments in parallel.

Prefer the cheapest diagnostic that distinguishes remaining hypotheses.

Distinguish:
    observation
    interpretation
    hypothesis
    falsifiable prediction
    next experiment

If evidence is ambiguous or a major architecture/physics decision is required, stop and request human review.

Never infer folds or physical branch behavior from unaudited states.

Failed prototypes stay isolated/default OFF.

The response must describe a collaboration decision only. Do not emit source patches, shell commands that mutate git history, or instructions to edit accepted checkpoints. Request specific additional artifact paths in requested_artifacts when the supplied evidence is insufficient. Do not request complete logs or the whole repository."""

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "classification": {"type": "string"},
        "experiment_count": {"type": "integer", "const": 1},
        "breakthrough": {"type": "boolean"},
        "human_review_required": {"type": "boolean"},
        "current_interpretation": {"type": "string"},
        "hypothesis": {"type": "string"},
        "prediction": {"type": "string"},
        "next_experiment": {"type": "string"},
        "forbidden_changes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "required_reporting": {
            "type": "array",
            "items": {"type": "string"},
        },
        "stop_condition": {"type": "string"},
        "requested_artifacts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "stop_reason": {"type": ["string", "null"]},
    },
    "required": [
        "classification",
        "experiment_count",
        "breakthrough",
        "human_review_required",
        "current_interpretation",
        "hypothesis",
        "prediction",
        "next_experiment",
        "forbidden_changes",
        "required_reporting",
        "stop_condition",
        "requested_artifacts",
        "stop_reason",
    ],
    "additionalProperties": False,
}

EXPECTED_DECISION_KEYS = frozenset(DECISION_SCHEMA["required"])


class BridgeError(RuntimeError):
    """A bounded bridge precondition or validation failure."""


@dataclass(frozen=True)
class GitState:
    branch: str
    head: str
    dirty: bool
    porcelain: str


@dataclass(frozen=True)
class ReasoningInput:
    text: str
    included_artifacts: tuple[str, ...]
    status_sha256: str
    byte_count: int


@dataclass(frozen=True)
class Decision:
    classification: str
    experiment_count: int
    breakthrough: bool
    human_review_required: bool
    current_interpretation: str
    hypothesis: str
    prediction: str
    next_experiment: str
    forbidden_changes: tuple[str, ...]
    required_reporting: tuple[str, ...]
    stop_condition: str
    requested_artifacts: tuple[str, ...]
    stop_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "experiment_count": self.experiment_count,
            "breakthrough": self.breakthrough,
            "human_review_required": self.human_review_required,
            "current_interpretation": self.current_interpretation,
            "hypothesis": self.hypothesis,
            "prediction": self.prediction,
            "next_experiment": self.next_experiment,
            "forbidden_changes": list(self.forbidden_changes),
            "required_reporting": list(self.required_reporting),
            "stop_condition": self.stop_condition,
            "requested_artifacts": list(self.requested_artifacts),
            "stop_reason": self.stop_reason,
        }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BridgeError(
            f"git {' '.join(arguments)} failed; repository state is inconsistent"
        )
    return result.stdout.strip()


def inspect_git_state(root: Path) -> GitState:
    branch = run_git(root, "branch", "--show-current")
    head = run_git(root, "rev-parse", "HEAD")
    porcelain = run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return GitState(
        branch=branch,
        head=head,
        dirty=bool(porcelain),
        porcelain=porcelain,
    )


def enforce_git_preconditions(state: GitState) -> None:
    if state.branch != EXPECTED_BRANCH:
        raise BridgeError(
            f"expected branch {EXPECTED_BRANCH!r}, found {state.branch!r}; "
            "qss-dcr must never be used by this wrapper"
        )
    if state.dirty:
        raise BridgeError(
            "working tree contains unexpected changes; stop and request human review"
        )


def bounded_environment_integer(name: str, default: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise BridgeError(f"{name} must be an integer") from error
    if value <= 0:
        raise BridgeError(f"{name} must be positive")
    if value > maximum:
        raise BridgeError(f"{name} must not exceed {maximum}")
    return value


def read_required_file(root: Path, relative_path: str) -> str:
    path = root / relative_path
    if not path.is_file():
        raise BridgeError(f"required collaboration file is missing: {relative_path}")
    return path.read_text(encoding="utf-8")


def explicit_small_report_paths(status: str) -> list[str]:
    candidates: list[str] = []
    for candidate in re.findall(r"`([^`]+\.(?:md|txt))`", status, flags=re.I):
        normalized = candidate.strip().replace("\\", "/")
        if normalized.startswith(".dcr-agent/"):
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def safe_report_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BridgeError(f"unsafe artifact path in STATUS.md: {relative_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BridgeError(f"artifact path escapes repository: {relative_path}") from error
    return resolved


def build_reasoning_input(
    root: Path,
    git_state: GitState,
    max_input_bytes: int,
    max_artifact_bytes: int,
) -> ReasoningInput:
    contents = {
        relative_path: read_required_file(root, relative_path)
        for relative_path in COLLABORATION_FILES
    }
    status = contents[".dcr-agent/STATUS.md"]
    status_sha256 = hashlib.sha256(status.encode("utf-8")).hexdigest()

    sections = [
        "GIT STATE",
        f"branch: {git_state.branch}",
        f"HEAD: {git_state.head}",
        f"working_tree: {'dirty' if git_state.dirty else 'clean'}",
        f"STATUS_sha256: {status_sha256}",
    ]
    for relative_path in COLLABORATION_FILES:
        sections.extend(
            [
                "",
                f"BEGIN {relative_path}",
                contents[relative_path],
                f"END {relative_path}",
            ]
        )

    included_artifacts: list[str] = []
    for relative_path in explicit_small_report_paths(status):
        path = safe_report_path(root, relative_path)
        if not path.is_file():
            raise BridgeError(
                f"STATUS.md explicitly lists a missing diagnostic report: {relative_path}"
            )
        report_bytes = path.read_bytes()
        if len(report_bytes) > max_artifact_bytes:
            raise BridgeError(
                f"explicit report exceeds DCR_AGENT_MAX_ARTIFACT_BYTES: {relative_path}"
            )
        try:
            report_text = report_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BridgeError(
                f"explicit report is not UTF-8 text: {relative_path}"
            ) from error
        sections.extend(
            [
                "",
                f"BEGIN REFERENCED REPORT {relative_path}",
                report_text,
                f"END REFERENCED REPORT {relative_path}",
            ]
        )
        included_artifacts.append(relative_path)

    text = "\n".join(sections)
    byte_count = len(text.encode("utf-8"))
    if byte_count > max_input_bytes:
        raise BridgeError(
            f"reasoning input is {byte_count} bytes, exceeding configured limit "
            f"{max_input_bytes}; no API call was made"
        )
    return ReasoningInput(
        text=text,
        included_artifacts=tuple(included_artifacts),
        status_sha256=status_sha256,
        byte_count=byte_count,
    )


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BridgeError(f"structured response field {field!r} must be nonempty text")
    return value.strip()


def require_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BridgeError(f"structured response field {field!r} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(require_nonempty_string(item, f"{field}[{index}]"))
    return tuple(result)


def validate_decision(payload: Mapping[str, Any]) -> Decision:
    if set(payload) != EXPECTED_DECISION_KEYS:
        missing = sorted(EXPECTED_DECISION_KEYS - set(payload))
        extra = sorted(set(payload) - EXPECTED_DECISION_KEYS)
        raise BridgeError(
            f"structured response keys do not match schema; missing={missing}, extra={extra}"
        )
    if type(payload["breakthrough"]) is not bool:
        raise BridgeError("structured response field 'breakthrough' must be boolean")
    if type(payload["human_review_required"]) is not bool:
        raise BridgeError(
            "structured response field 'human_review_required' must be boolean"
        )
    if type(payload["experiment_count"]) is not int or payload["experiment_count"] != 1:
        raise BridgeError("structured response must authorize exactly one experiment")
    stop_reason_value = payload["stop_reason"]
    if stop_reason_value is not None:
        stop_reason_value = require_nonempty_string(stop_reason_value, "stop_reason")

    decision = Decision(
        classification=require_nonempty_string(
            payload["classification"], "classification"
        ),
        experiment_count=payload["experiment_count"],
        breakthrough=payload["breakthrough"],
        human_review_required=payload["human_review_required"],
        current_interpretation=require_nonempty_string(
            payload["current_interpretation"], "current_interpretation"
        ),
        hypothesis=require_nonempty_string(payload["hypothesis"], "hypothesis"),
        prediction=require_nonempty_string(payload["prediction"], "prediction"),
        next_experiment=require_nonempty_string(
            payload["next_experiment"], "next_experiment"
        ),
        forbidden_changes=require_string_list(
            payload["forbidden_changes"], "forbidden_changes"
        ),
        required_reporting=require_string_list(
            payload["required_reporting"], "required_reporting"
        ),
        stop_condition=require_nonempty_string(
            payload["stop_condition"], "stop_condition"
        ),
        requested_artifacts=require_string_list(
            payload["requested_artifacts"], "requested_artifacts"
        ),
        stop_reason=stop_reason_value,
    )
    if not decision.forbidden_changes:
        raise BridgeError("structured response must name forbidden changes")
    if not decision.required_reporting:
        raise BridgeError("structured response must name required reporting")
    return decision


def parse_response_output(response: Any) -> Mapping[str, Any]:
    if getattr(response, "status", None) != "completed":
        raise BridgeError("Responses API result was incomplete; no retry was attempted")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise BridgeError("Responses API returned no structured output text")
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise BridgeError("Responses API output was not valid JSON") from error
    if not isinstance(payload, dict):
        raise BridgeError("Responses API output must be a JSON object")
    return payload


def call_reasoning_api(model: str, reasoning_input: ReasoningInput) -> Decision:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise BridgeError("OPENAI_API_KEY is not set; no API call was made")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise BridgeError(
            "official OpenAI SDK is unavailable; run: python -m pip install openai"
        ) from error

    client = OpenAI(max_retries=0)
    try:
        response = client.responses.create(
            model=model,
            instructions=PERMANENT_INSTRUCTION,
            input=reasoning_input.text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dcr_reasoning_decision",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                }
            },
        )
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        if status_code in {400, 404}:
            raise BridgeError(
                f"configured model {model!r} is unavailable or incompatible; "
                "no fallback model was selected"
            ) from error
        raise BridgeError(
            f"Responses API request failed for configured model {model!r} "
            f"with {type(error).__name__}; no fallback model was selected"
        ) from error
    return validate_decision(parse_response_output(response))


def previous_api_classifications(history: str) -> list[str]:
    return re.findall(
        r"^- API classification: `([^`]+)`$", history, flags=re.MULTILINE
    )


def hard_stop_reason(decision: Decision, history: str) -> str | None:
    if decision.breakthrough:
        return "reasoning result marked a breakthrough"
    if decision.human_review_required:
        return decision.stop_reason or "reasoning result requires human review"
    if decision.requested_artifacts:
        return "reasoning model requested additional artifact content"
    if decision.stop_reason is not None:
        return decision.stop_reason

    classification = decision.classification.lower()
    ambiguity_text = " ".join(
        (
            decision.classification,
            decision.current_interpretation,
            decision.hypothesis,
            decision.prediction,
        )
    ).lower()
    if any(word in ambiguity_text for word in ("ambiguous", "uncertain", "unknown")):
        return "reasoning classification is ambiguous"

    proposed = " ".join((decision.hypothesis, decision.next_experiment)).lower()
    proposed_actions = re.sub(
        r"\b(?:do not|don't|without|no)\s+"
        r"(?:change|changing|modify|modifying|alter|altering|replace|replacing)\b"
        r".{0,80}\b(?:physical equations?|reaction rates?|boundary conditions?)\b",
        "",
        proposed,
    )
    forbidden_patterns = (
        (r"pseudo[- ]arclength", "pseudo-arclength was proposed"),
        (r"monolithic", "monolithic architecture was proposed"),
        (r"newton architecture", "new Newton architecture was proposed"),
        (r"overwrite.{0,50}accepted.{0,30}checkpoint", "accepted checkpoint overwrite was proposed"),
        (r"production checkpoint candidate", "production checkpoint candidate requires review"),
        (r"git reset --hard|force[- ]push|git push --force", "irreversible repository action was proposed"),
        (
            r"\b(?:change|modify|alter|replace)\b.{0,80}"
            r"\b(?:physical equations?|governing equations?|physical model)\b",
            "physical-equation change was proposed",
        ),
        (
            r"\b(?:change|modify|alter|replace)\b.{0,80}"
            r"\b(?:reaction rates?|rate coefficients?)\b",
            "reaction-rate change was proposed",
        ),
        (
            r"\b(?:change|modify|alter|replace)\b.{0,80}"
            r"\b(?:boundary conditions?|boundary physics)\b",
            "boundary-condition physics change was proposed",
        ),
    )
    for pattern, reason in forbidden_patterns:
        if re.search(pattern, proposed_actions):
            return reason

    evidence = " ".join(
        (decision.current_interpretation, decision.hypothesis, decision.prediction)
    ).lower()
    if re.search(
        r"(?:evidence (?:for|of)|indicates?|confirms?|suggests?).{0,80}"
        r"(?:fold|multiple physical branches?|multiple solution branches?)",
        evidence,
    ):
        return "branch or fold evidence requires human review"
    if "production checkpoint candidate" in evidence:
        return "production checkpoint candidate requires human review"

    previous = previous_api_classifications(history)
    if previous and previous[-1].strip().lower() == classification.strip():
        return "same failure classification repeated for two consecutive cycles"

    normalized_experiment = decision.next_experiment.lower()
    if normalized_experiment in {"none", "n/a", "no experiment"}:
        return "no single discriminating experiment was identified"
    if re.search(r"\n\s*(?:2[.)]|second experiment\b)", decision.next_experiment, re.I):
        return "multiple experiments were proposed"
    if re.search(r"\b(?:in parallel|alternatively)\b", decision.next_experiment, re.I):
        return "multiple experiments were proposed"
    if re.search(
        r"\b(?:then|afterwards|subsequently|next)\s+"
        r"(?:run|execute|test|try|perform|repeat)\b",
        decision.next_experiment,
        re.I,
    ):
        return "multiple sequential experiments were proposed"
    if re.search(
        r"\band also\s+(?:run|execute|test|try|perform|repeat)\b",
        decision.next_experiment,
        re.I,
    ):
        return "multiple experiments were proposed"
    if re.search(r"\bfirst\b.{0,160}\bsecond\b", decision.next_experiment, re.I):
        return "multiple experiments were proposed"
    return None


def render_chatgpt_next(decision: Decision) -> str:
    forbidden = "\n".join(f"- {item}" for item in decision.forbidden_changes)
    reporting = "\n".join(f"- {item}" for item in decision.required_reporting)
    return (
        "# CURRENT INTERPRETATION\n\n"
        f"{decision.current_interpretation}\n\n"
        "# HYPOTHESIS\n\n"
        f"{decision.hypothesis}\n\n"
        "# FALSIFIABLE PREDICTION\n\n"
        f"{decision.prediction}\n\n"
        "# EXACTLY ONE NEXT EXPERIMENT\n\n"
        f"{decision.next_experiment}\n\n"
        "# WHAT MUST NOT CHANGE\n\n"
        f"{forbidden}\n\n"
        "# REQUIRED REPORTING\n\n"
        f"{reporting}\n\n"
        "# STOP CONDITION\n\n"
        f"{decision.stop_condition}\n"
    )


def render_attention(decision: Decision, reason: str) -> str:
    return (
        "# Human Attention Required\n\n"
        f"- Classification: `{decision.classification}`\n"
        f"- Reason: {reason}\n"
        f"- Interpretation: {decision.current_interpretation}\n"
        f"- Requested artifacts: "
        f"{', '.join(decision.requested_artifacts) or 'none'}\n"
    )


def next_cycle_number(history: str) -> int:
    cycles = [int(value) for value in re.findall(r"^## API reasoning cycle (\d+)$", history, re.M)]
    return max(cycles, default=0) + 1


def render_history_entry(
    cycle: int,
    timestamp: str,
    git_state: GitState,
    reasoning_input: ReasoningInput,
    decision: Decision,
    stop_reason: str | None,
) -> str:
    experiment = decision.next_experiment if stop_reason is None else "not authorized"
    return (
        f"\n## API reasoning cycle {cycle}\n\n"
        f"- Timestamp: `{timestamp}`\n"
        f"- Input branch: `{git_state.branch}`\n"
        f"- Input HEAD: `{git_state.head}`\n"
        f"- Input git state: `{'dirty' if git_state.dirty else 'clean'}`\n"
        f"- Input STATUS SHA-256: `{reasoning_input.status_sha256}`\n"
        f"- API classification: `{decision.classification}`\n"
        f"- API interpretation: {decision.current_interpretation}\n"
        f"- Hypothesis: {decision.hypothesis}\n"
        f"- Selected next experiment: {experiment}\n"
        f"- Human review required: `{'yes' if stop_reason else 'no'}`\n"
        f"- Stop reason: {stop_reason or 'none'}\n"
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def apply_decision(
    root: Path,
    git_state: GitState,
    reasoning_input: ReasoningInput,
    decision: Decision,
    stop_reason: str | None,
) -> None:
    history_path = root / ".dcr-agent/HISTORY.md"
    history = history_path.read_text(encoding="utf-8")
    cycle = next_cycle_number(history)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = render_history_entry(
        cycle,
        timestamp,
        git_state,
        reasoning_input,
        decision,
        stop_reason,
    )
    atomic_write(history_path, history.rstrip() + "\n" + entry)

    if stop_reason is None:
        atomic_write(
            root / ".dcr-agent/CHATGPT_NEXT.md",
            render_chatgpt_next(decision),
        )
        attention = root / ".dcr-agent/ATTENTION.md"
        if attention.exists():
            attention.unlink()
    else:
        atomic_write(
            root / ".dcr-agent/CHATGPT_NEXT.md",
            "# BLOCKED\n\nNo experiment is authorized. See `ATTENTION.md`.\n",
        )
        atomic_write(
            root / ".dcr-agent/ATTENTION.md",
            render_attention(decision, stop_reason),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded DCR Responses API reasoning bridge",
        epilog=(
            "Install dependency: python -m pip install openai. "
            "The SDK reads OPENAI_API_KEY from the environment."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="call and validate the API but do not edit, execute, commit, or push",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        metavar="N",
        help="safety cap for this invocation (1-3; version one stops after one handoff)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.max_cycles < 1 or arguments.max_cycles > MAX_CYCLES_CAP:
        raise BridgeError(
            f"--max-cycles must be between 1 and {MAX_CYCLES_CAP}"
        )

    root = repository_root()
    git_state = inspect_git_state(root)
    enforce_git_preconditions(git_state)
    model = os.environ.get("DCR_REASONING_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise BridgeError("DCR_REASONING_MODEL must not be empty")
    max_input_bytes = bounded_environment_integer(
        "DCR_AGENT_MAX_INPUT_BYTES",
        DEFAULT_MAX_INPUT_BYTES,
        MAX_INPUT_BYTES_CAP,
    )
    max_artifact_bytes = bounded_environment_integer(
        "DCR_AGENT_MAX_ARTIFACT_BYTES",
        DEFAULT_MAX_ARTIFACT_BYTES,
        MAX_ARTIFACT_BYTES_CAP,
    )
    reasoning_input = build_reasoning_input(
        root,
        git_state,
        max_input_bytes,
        max_artifact_bytes,
    )
    history = read_required_file(root, ".dcr-agent/HISTORY.md")
    decision = call_reasoning_api(model, reasoning_input)
    stop_reason = hard_stop_reason(decision, history)
    proposed_next = render_chatgpt_next(decision)

    print(f"API model: {model}")
    print(f"Git branch: {git_state.branch}")
    print(f"Git HEAD: {git_state.head}")
    print("Git state: clean")
    print(f"Configured max cycles: {arguments.max_cycles} (one handoff executed)")
    print(f"Input bytes: {reasoning_input.byte_count}")
    print(
        "Included reports: "
        + (", ".join(reasoning_input.included_artifacts) or "none")
    )
    print("Structured API result:")
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    if stop_reason:
        print(f"HARD STOP: {stop_reason}")
        print("No CHATGPT_NEXT experiment is authorized.")
    else:
        print("Proposed CHATGPT_NEXT.md:")
        print(proposed_next)

    if arguments.dry_run:
        print("DRY RUN: no files edited; no Codex call, solver run, commit, or push")
        return 0

    current_git_state = inspect_git_state(root)
    enforce_git_preconditions(current_git_state)
    if current_git_state.head != git_state.head:
        raise BridgeError(
            "git HEAD changed during the API request; stop and request human review"
        )
    apply_decision(
        root,
        git_state,
        reasoning_input,
        decision,
        stop_reason,
    )
    if stop_reason:
        print("ATTENTION.md written; no experiment was authorized")
        return 2
    print("One reasoning handoff written; external Codex execution is required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as error:
        print(f"DCR agent bridge stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
