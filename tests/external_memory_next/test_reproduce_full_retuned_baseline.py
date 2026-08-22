from experiments.external_memory_next.reproduce_full_retuned_baseline_v1 import final_rerank, merge_stage1


def test_frozen_stage1_and_stage2_arithmetic() -> None:
    feature = {
        "generic_frequency_candidates": [
            {"candidate": "甲", "source": "generic_frequency", "generic_rank": 1,
             "normalized_generic_score": 1.0, "final_score": 1.0},
            {"candidate": "乙", "source": "generic_frequency", "generic_rank": 2,
             "normalized_generic_score": 0.5, "final_score": 0.5},
        ],
        "personal_k5": ["丙"], "p_ng": {"丙": .25}, "choice_share": {"丙": .5},
        "entropy_concentration": .25,
    }
    weights = {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0, "lambda_n": 6.0, "lambda_b": 6.0}
    stage1 = merge_stage1(feature, weights)
    assert [row["candidate"] for row in stage1] == ["丙", "甲", "乙"]
    assert stage1[0]["final_score"] == 5.0
    final = final_rerank(stage1, {"丙": 0, "甲": 1, "乙": 0},
                         {"丙": 0, "甲": 0, "乙": 0}, weights)
    assert [row["candidate"] for row in final] == ["甲", "丙", "乙"]
