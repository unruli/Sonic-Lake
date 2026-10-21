from scripts.evaluate_clap_events import class_prompt


def test_class_prompt_is_frozen_and_human_readable() -> None:
    assert class_prompt("clock_alarm") == "a recording of the sound of clock alarm"
