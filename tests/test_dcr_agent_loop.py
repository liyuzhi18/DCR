#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/dcr_agent_loop.py"
SPEC = importlib.util.spec_from_file_location("dcr_agent_loop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dcr_agent_loop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dcr_agent_loop
SPEC.loader.exec_module(dcr_agent_loop)


def valid_payload() -> dict:
    return {
        "classification": "detector_false_positive",
        "experiment_count": 1,
        "breakthrough": False,
        "human_review_required": False,
        "current_interpretation": "The conservation projection is mis-scaled.",
        "hypothesis": "A row-scale-aware diagnostic will remove only the false stop.",
        "prediction": "The replay reaches the next existing stopping condition.",
        "next_experiment": "Replay once with a read-only alternative detector scale.",
        "forbidden_changes": [
            "Do not change physical equations.",
            "Do not overwrite accepted checkpoints.",
        ],
        "required_reporting": ["Report both old and alternative detector values."],
        "stop_condition": "Stop after the first divergence or existing solver stop.",
        "requested_artifacts": [],
        "stop_reason": None,
    }


class DecisionValidationTests(unittest.TestCase):
    def test_valid_decision_round_trips(self) -> None:
        decision = dcr_agent_loop.validate_decision(valid_payload())
        self.assertEqual(decision.classification, "detector_false_positive")
        self.assertIsNone(decision.stop_reason)
        self.assertIn(
            "# EXACTLY ONE NEXT EXPERIMENT",
            dcr_agent_loop.render_chatgpt_next(decision),
        )

    def test_missing_key_stops_without_retry(self) -> None:
        payload = valid_payload()
        del payload["prediction"]
        with self.assertRaises(dcr_agent_loop.BridgeError):
            dcr_agent_loop.validate_decision(payload)

    def test_wrong_boolean_type_stops(self) -> None:
        payload = valid_payload()
        payload["breakthrough"] = "false"
        with self.assertRaises(dcr_agent_loop.BridgeError):
            dcr_agent_loop.validate_decision(payload)


class SafetyTests(unittest.TestCase):
    def test_dirty_git_state_stops(self) -> None:
        state = dcr_agent_loop.GitState(
            branch=dcr_agent_loop.EXPECTED_BRANCH,
            head="abc",
            dirty=True,
            porcelain="?? file",
        )
        with self.assertRaises(dcr_agent_loop.BridgeError):
            dcr_agent_loop.enforce_git_preconditions(state)

    def test_qss_branch_stops(self) -> None:
        state = dcr_agent_loop.GitState(
            branch="qss-dcr", head="abc", dirty=False, porcelain=""
        )
        with self.assertRaises(dcr_agent_loop.BridgeError):
            dcr_agent_loop.enforce_git_preconditions(state)

    def test_major_architecture_proposal_stops(self) -> None:
        payload = valid_payload()
        payload["next_experiment"] = "Implement a monolithic solver."
        decision = dcr_agent_loop.validate_decision(payload)
        reason = dcr_agent_loop.hard_stop_reason(decision, "")
        self.assertIn("monolithic", reason)

    def test_explicit_non_change_does_not_trigger_physics_stop(self) -> None:
        payload = valid_payload()
        payload["next_experiment"] = (
            "Replay the detector once without changing physical equations."
        )
        decision = dcr_agent_loop.validate_decision(payload)
        self.assertIsNone(dcr_agent_loop.hard_stop_reason(decision, ""))

    def test_ambiguous_interpretation_stops(self) -> None:
        payload = valid_payload()
        payload["current_interpretation"] = "The evidence remains ambiguous."
        decision = dcr_agent_loop.validate_decision(payload)
        self.assertIn(
            "ambiguous", dcr_agent_loop.hard_stop_reason(decision, "")
        )

    def test_sequential_experiments_stop(self) -> None:
        payload = valid_payload()
        payload["next_experiment"] = "Run replay A, then run replay B."
        decision = dcr_agent_loop.validate_decision(payload)
        self.assertIn(
            "sequential", dcr_agent_loop.hard_stop_reason(decision, "")
        )

    def test_and_also_experiments_stop(self) -> None:
        payload = valid_payload()
        payload["next_experiment"] = "Run replay A and also execute replay B."
        decision = dcr_agent_loop.validate_decision(payload)
        self.assertIn(
            "multiple", dcr_agent_loop.hard_stop_reason(decision, "")
        )

    def test_experiment_count_must_be_one(self) -> None:
        payload = valid_payload()
        payload["experiment_count"] = 2
        with self.assertRaises(dcr_agent_loop.BridgeError):
            dcr_agent_loop.validate_decision(payload)

    def test_requested_artifact_stops_first_version(self) -> None:
        payload = valid_payload()
        payload["requested_artifacts"] = ["path/to/report.md"]
        decision = dcr_agent_loop.validate_decision(payload)
        self.assertIsNotNone(dcr_agent_loop.hard_stop_reason(decision, ""))

    def test_repeated_classification_stops(self) -> None:
        decision = dcr_agent_loop.validate_decision(valid_payload())
        history = "- API classification: `detector_false_positive`\n"
        reason = dcr_agent_loop.hard_stop_reason(decision, history)
        self.assertIn("repeated", reason)


class InputTests(unittest.TestCase):
    def test_environment_input_limit_has_hard_cap(self) -> None:
        with mock.patch.dict(
            "os.environ", {"DCR_AGENT_MAX_INPUT_BYTES": "500001"}
        ):
            with self.assertRaises(dcr_agent_loop.BridgeError):
                dcr_agent_loop.bounded_environment_integer(
                    "DCR_AGENT_MAX_INPUT_BYTES",
                    dcr_agent_loop.DEFAULT_MAX_INPUT_BYTES,
                    dcr_agent_loop.MAX_INPUT_BYTES_CAP,
                )

    def test_only_explicit_small_text_reports_are_included(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / ".dcr-agent"
            agent.mkdir()
            (agent / "RULES.md").write_text("rules", encoding="utf-8")
            (agent / "HISTORY.md").write_text("history", encoding="utf-8")
            (agent / "CHATGPT_NEXT.md").write_text("previous", encoding="utf-8")
            (agent / "STATUS.md").write_text(
                "- Diagnostic report: `reports/small.md`\n"
                "- Run log: `reports/huge.log`\n",
                encoding="utf-8",
            )
            reports = root / "reports"
            reports.mkdir()
            (reports / "small.md").write_text("small report", encoding="utf-8")
            (reports / "huge.log").write_text("not included", encoding="utf-8")
            state = dcr_agent_loop.GitState(
                branch=dcr_agent_loop.EXPECTED_BRANCH,
                head="abc",
                dirty=False,
                porcelain="",
            )
            result = dcr_agent_loop.build_reasoning_input(
                root, state, max_input_bytes=10_000, max_artifact_bytes=1_000
            )
            self.assertEqual(result.included_artifacts, ("reports/small.md",))
            self.assertIn("small report", result.text)
            self.assertNotIn("not included", result.text)

    def test_input_limit_stops_before_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / ".dcr-agent"
            agent.mkdir()
            for name in ("RULES.md", "STATUS.md", "HISTORY.md", "CHATGPT_NEXT.md"):
                (agent / name).write_text("x" * 100, encoding="utf-8")
            state = dcr_agent_loop.GitState(
                branch=dcr_agent_loop.EXPECTED_BRANCH,
                head="abc",
                dirty=False,
                porcelain="",
            )
            with self.assertRaises(dcr_agent_loop.BridgeError):
                dcr_agent_loop.build_reasoning_input(
                    root, state, max_input_bytes=10, max_artifact_bytes=1_000
                )


class FileTransitionTests(unittest.TestCase):
    def test_hard_stop_blocks_previous_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / ".dcr-agent"
            agent.mkdir()
            (agent / "HISTORY.md").write_text("# History\n", encoding="utf-8")
            (agent / "CHATGPT_NEXT.md").write_text(
                "old authorized experiment", encoding="utf-8"
            )
            decision = dcr_agent_loop.validate_decision(valid_payload())
            state = dcr_agent_loop.GitState(
                branch=dcr_agent_loop.EXPECTED_BRANCH,
                head="abc",
                dirty=False,
                porcelain="",
            )
            reasoning_input = dcr_agent_loop.ReasoningInput(
                text="input",
                included_artifacts=(),
                status_sha256="hash",
                byte_count=5,
            )
            dcr_agent_loop.apply_decision(
                root,
                state,
                reasoning_input,
                decision,
                "human review",
            )
            self.assertIn(
                "No experiment is authorized",
                (agent / "CHATGPT_NEXT.md").read_text(encoding="utf-8"),
            )
            self.assertTrue((agent / "ATTENTION.md").is_file())


class ApiBoundaryTests(unittest.TestCase):
    def test_sdk_retries_are_disabled(self) -> None:
        captured: dict = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured["request"] = kwargs
                return types.SimpleNamespace(
                    status="completed", output_text=json.dumps(valid_payload())
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.responses = FakeResponses()

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        reasoning_input = dcr_agent_loop.ReasoningInput(
            text="input",
            included_artifacts=(),
            status_sha256="hash",
            byte_count=5,
        )
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
            sys.modules, {"openai": fake_module}
        ):
            decision = dcr_agent_loop.call_reasoning_api(
                "configured-model", reasoning_input
            )
        self.assertEqual(captured["client"], {"max_retries": 0})
        self.assertEqual(captured["request"]["model"], "configured-model")
        self.assertEqual(decision.classification, "detector_false_positive")


if __name__ == "__main__":
    unittest.main()
