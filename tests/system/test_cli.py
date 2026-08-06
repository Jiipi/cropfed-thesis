import argparse
import io
import unittest
from pathlib import Path

from cropfed.cli import _configure_utf8_stream, _research_result_valid, build_parser


class CliOutputTests(unittest.TestCase):
    def test_reconfigures_legacy_windows_stream_for_vietnamese_output(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1258")

        _configure_utf8_stream(stream)
        stream.write("Phát hiện sâu bệnh: khỏe")
        stream.flush()

        self.assertEqual(stream.encoding.lower(), "utf-8")
        self.assertEqual(
            buffer.getvalue().decode("utf-8"),
            "Phát hiện sâu bệnh: khỏe",
        )


    def test_training_runs_are_unclassified_by_default(self) -> None:
        args = build_parser().parse_args(["train-centralized"])

        self.assertIsNone(_research_result_valid(args))

    def test_research_run_requires_a_protocol_lock(self) -> None:
        args = argparse.Namespace(research_run=True, pilot=False, protocol_lock=None)

        with self.assertRaisesRegex(ValueError, "--protocol-lock"):
            _research_result_valid(args)

        args.protocol_lock = Path("configs/protocol_lock.json")
        self.assertTrue(_research_result_valid(args))

    def test_pilot_and_research_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["train-local-only", "--pilot", "--research-run"]
            )


if __name__ == "__main__":
    unittest.main()
