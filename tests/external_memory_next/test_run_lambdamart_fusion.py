from experiments.external_memory_next.run_lambdamart_fusion_v1 import configurations, ranks_from_scores, selection_key


def test_grid_matches_predeclared_design() -> None:
    configs = configurations()
    assert len(configs) == 13
    assert sum(config["kind"] == "nonlinear" for config in configs) == 12


def test_score_ranking_uses_frozen_order_as_tiebreak() -> None:
    meta = [{"row_id": "r", "gold": "乙", "offset": 0, "candidate_count": 2,
             "candidates": ["甲", "乙"], "baseline_top10": ["乙", "甲"]}]
    ranks, top10 = ranks_from_scores([.5, .5], meta)
    assert ranks == [1]
    assert top10 == [["乙", "甲"]]


def test_selection_uses_macro_before_complexity() -> None:
    def record(macro, depth):
        return {"config_id": str(depth), "config": {"max_depth": depth, "rounds": 50, "min_data_in_leaf": 100},
                "metrics": {"overall": {"macro_author_top1": macro, "micro_top1": .8, "mrr_at_10": .8}}}
    assert selection_key(record(.81, 5)) < selection_key(record(.80, 2))
