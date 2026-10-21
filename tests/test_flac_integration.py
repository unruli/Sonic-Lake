from scripts.build_production_profiles import REPRESENTATIONS as PROFILE_REPRESENTATIONS
from scripts.evaluate_heldout_design import REPRESENTATIONS as HELDOUT_REPRESENTATIONS
from scripts.make_paper_artifacts import display_name


def test_flac_is_available_to_profile_and_heldout_analysis() -> None:
    assert PROFILE_REPRESENTATIONS["flac"] == "flac_lossless_c8"
    assert HELDOUT_REPRESENTATIONS["flac"] == "flac_lossless_c8"


def test_flac_has_a_paper_display_name() -> None:
    assert display_name("flac@lossless_c8") == "FLAC lossless_c8"
