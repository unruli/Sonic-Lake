import unittest

from soniclake.optimizer import optimize
from soniclake.schema import FamilyProfile, RepresentationProfile


def profile(name, storage, lexical_quality, event_quality, lexical_cost=1.0, event_cost=1.0):
    return RepresentationProfile(
        name=name,
        storage_bytes_per_hour=storage,
        construction_s_per_hour=1.0,
        families={
            "lexical": FamilyProfile(lexical_quality, lexical_cost),
            "acoustic_event": FamilyProfile(event_quality, event_cost),
        },
    )


class OptimizerTests(unittest.TestCase):
    def test_materializes_complementary_representations(self):
        profiles = {
            "text": profile("text", 10, 0.95, 0.1),
            "audio": profile("audio", 30, 0.4, 0.95),
            "raw": profile("raw", 100, 0.99, 0.99),
        }
        design = optimize(
            profiles,
            frequencies={"lexical": 0.5, "acoustic_event": 0.5},
            thresholds={"lexical": 0.9, "acoustic_event": 0.9},
            storage_weight=1.0,
            construction_weight=0.0,
            query_weight=0.0,
        )
        self.assertEqual(design.representations, ("audio", "text"))
        self.assertEqual(design.assignment, {"lexical": "text", "acoustic_event": "audio"})

    def test_rejects_infeasible_budget(self):
        profiles = {"text": profile("text", 10, 0.95, 0.1)}
        with self.assertRaisesRegex(ValueError, "no feasible"):
            optimize(
                profiles,
                frequencies={"lexical": 0.5, "acoustic_event": 0.5},
                thresholds={"lexical": 0.9, "acoustic_event": 0.9},
            )

    def test_tie_break_does_not_materialize_unused_representation(self):
        profiles = {
            "compact": profile("compact", 10, 0.95, 0.95),
            "extra": profile("extra", 20, 0.95, 0.95),
        }
        design = optimize(
            profiles,
            frequencies={"lexical": 0.5, "acoustic_event": 0.5},
            thresholds={"lexical": 0.9, "acoustic_event": 0.9},
            storage_weight=0.0,
            construction_weight=0.0,
            query_weight=0.0,
        )
        self.assertEqual(design.representations, ("compact",))
