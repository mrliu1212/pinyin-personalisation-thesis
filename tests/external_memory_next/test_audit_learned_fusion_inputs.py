from experiments.external_memory_next.audit_learned_fusion_inputs_v1 import FEATURE_NAMES, extract_group


def test_feature_extractor_separates_runtime_features_from_label() -> None:
    feature = {"row_id": "r", "author": "a", "gold": "甲",
               "entropy_concentration": .5, "same_pinyin_history_count": 2,
               "raw_history_count": 3}
    support = {"row_id": "r", "retuned_stage1_candidates": [
        {"candidate": "甲", "source": "generic_frequency", "final_score": 2,
         "base_rank": 1, "generic_rank": 1, "generic_score": -1,
         "normalized_generic_score": 1, "frequency_count": 1, "personal_score": .5},
        {"candidate": "乙", "source": "personal_recovery", "final_score": 1,
         "base_rank": 2, "personal_candidate_rank": 1, "p_ng": .7, "choice_share": .5},
    ], "retuned_ngram_support": {"甲": .2, "乙": .8},
        "retuned_bge_support": {"甲": .3, "乙": .7},
        "bge_history_counts": {"甲": 1, "乙": 2},
        "ngram_effective_n": 1, "ngram_matched_history_rows": 2}
    vectors, labels, names = extract_group(feature, support)
    assert names == ["甲", "乙"]
    assert labels == [1, 0]
    assert len(vectors) == 2
    assert all(len(vector) == len(FEATURE_NAMES) for vector in vectors)
    assert "author" not in FEATURE_NAMES
    assert "gold" not in FEATURE_NAMES
