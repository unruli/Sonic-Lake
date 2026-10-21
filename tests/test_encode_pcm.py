from pathlib import Path

from scripts.encode_pcm import ffmpeg_command


def test_pcm_command_preserves_source_rate_and_channels() -> None:
    command = ffmpeg_command("ffmpeg", "input.flac", Path("output.wav"))
    assert command[-3:] == ["-c:a", "pcm_s16le", "output.wav"]
    assert "-ar" not in command
    assert "-ac" not in command
