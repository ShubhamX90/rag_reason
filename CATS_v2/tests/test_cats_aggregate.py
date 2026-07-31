import unittest

from rag_eval.evaluator import aggregate_sample_results


def sample(**overrides):
    row = {
        "sample_id": "fixture",
        "conflict_type": 1,
        "pred_answered": True,
        "gold_answerable": True,
        "correct_refusal": False,
        "gr_accuracy": 1.0,
        "behavior_score": 1.0,
        "behavior_applicable": True,
        "behavior_details": {
            "committee_details": {
                "weighted_for": 1.0,
                "weighted_against": 0.0,
                "votes_for": 3,
                "total_votes": 3,
            }
        },
        "factual_grounding_score": 0.25,
        "factual_grounding_applicable": True,
        "single_truth_recall_score": 0.0,
        "single_truth_applicable": False,
    }
    row.update(overrides)
    return row


class CatsAggregateTests(unittest.TestCase):
    def test_answerable_score_uses_gated_harmonic_mean(self):
        overall, _, _ = aggregate_sample_results([sample()])
        self.assertAlmostEqual(overall["cats_prevalence_score"], 0.4)

    def test_wrong_decision_is_a_zero_gate(self):
        overall, _, _ = aggregate_sample_results([sample(gr_accuracy=0.0)])
        self.assertEqual(overall["cats_prevalence_score"], 0.0)

    def test_correct_refusal_is_decision_only(self):
        overall, _, _ = aggregate_sample_results([
            sample(
                conflict_type=2,
                pred_answered=False,
                gold_answerable=False,
                correct_refusal=True,
                behavior_applicable=False,
                behavior_score=0.0,
                behavior_details={"skipped": "correct_refusal"},
                factual_grounding_applicable=False,
                single_truth_applicable=False,
            )
        ])
        self.assertAlmostEqual(overall["cats_prevalence_score"], 1.0)
        self.assertTrue(overall["cats_complete"])

    def test_committee_disagreement_is_preserved(self):
        overall, _, _ = aggregate_sample_results([
            sample(
                behavior_details={
                    "committee_details": {
                        "weighted_for": 2.0,
                        "weighted_against": 1.0,
                        "votes_for": 2,
                        "total_votes": 3,
                    }
                }
            )
        ])
        self.assertAlmostEqual(overall["behavior_consensus"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
