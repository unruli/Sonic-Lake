import tempfile
import unittest
import wave
from pathlib import Path

from soniclake.datasets import crema_subset_manifest, esc50_manifest


def write_silence(path: Path, frames: int = 800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * frames)


class DatasetTests(unittest.TestCase):
    def test_esc50_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            write_silence(root / "audio" / "1-test.wav")
            (root / "meta").mkdir()
            (root / "meta" / "esc50.csv").write_text(
                "filename,fold,target,category,esc10,src_file,take\n"
                "1-test.wav,1,3,dog,True,source.wav,A\n",
                encoding="utf-8",
            )
            samples = esc50_manifest(root)
        self.assertEqual(samples[0].split, "fold1")
        self.assertEqual(samples[0].labels["category"], "dog")
        self.assertEqual(samples[0].labels["group_id"], "source.wav")

    def test_crema_subset_label(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            write_silence(root / "cremad_00001_happy.wav")
            samples = crema_subset_manifest(root)
        self.assertEqual(samples[0].labels["emotion"], "happy")
        self.assertNotIn("actor_id", samples[0].labels)

    def test_crema_subset_actor_map(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            write_silence(root / "cremad_00001_happy.wav")
            actor_map = root / "actors.csv"
            actor_map.write_text(
                "recording_id,actor_id\ncremad_00001_happy,1001\n",
                encoding="utf-8",
            )
            samples = crema_subset_manifest(root, actor_map=actor_map)
        self.assertEqual(samples[0].labels["actor_id"], "1001")

    def test_crema_subset_rejects_incomplete_actor_map(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            write_silence(root / "cremad_00001_happy.wav")
            actor_map = root / "actors.csv"
            actor_map.write_text("recording_id,actor_id\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no entry"):
                crema_subset_manifest(root, actor_map=actor_map)
