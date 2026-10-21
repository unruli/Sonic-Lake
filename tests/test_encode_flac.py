from pathlib import Path

from scripts.encode_flac import ffmpeg_command


def test_flac_command_is_lossless_and_preserves_source_shape() -> None:
    command = ffmpeg_command("ffmpeg", "input.wav", Path("output.flac"), 8)
    assert command[-5:] == ["-c:a", "flac", "-compression_level", "8", "output.flac"]
    assert "-ar" not in command
    assert "-ac" not in command
