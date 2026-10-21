import tempfile
import unittest
from pathlib import Path

from soniclake.tokenpack import read_token_matrix, write_token_matrix


class TokenPackTests(unittest.TestCase):
    def test_round_trip(self):
        matrix = [[0, 1, 1023, 7], [10, 11, 12, 13]]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "codes.snct"
            header = write_token_matrix(
                path,
                matrix,
                vocab_size=1024,
                sample_rate=24000,
                frame_rate_hz=75,
                codec="test",
                model_revision="abc123",
            )
            loaded_header, loaded = read_token_matrix(path)
        self.assertEqual(header, loaded_header)
        self.assertEqual(matrix, loaded)
        self.assertEqual(header["bits_per_token"], 10)
