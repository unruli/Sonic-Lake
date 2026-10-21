from scripts.evaluate_emotion_prosody import FEATURE_NAMES


def test_feature_contract_is_frozen() -> None:
    assert len(FEATURE_NAMES) == 37
    assert FEATURE_NAMES[:3] == ["f0_mean", "f0_std", "voiced_frac"]
    assert FEATURE_NAMES[-1] == "mfcc12_std"
