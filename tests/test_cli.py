import io
import unittest

from cropfed.cli import _configure_utf8_stream


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


if __name__ == "__main__":
    unittest.main()
