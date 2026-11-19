"""A unit whose result has no boosted-tree or neural model is failed (core env)."""

from afe.run import checks


def test_trees_only_is_rejected():
    rf_et = {"RandomForestGini_BAG_L1": {}, "ExtraTrees_r42_BAG_L1": {}, "WeightedEnsemble_L2": {}}
    assert checks.verdict({"models": rf_et})


def test_each_result_format_with_a_boosted_or_nn_model_passes():
    assert checks.verdict({"models": {"RandomForest_BAG_L1": {}, "LightGBM_r13": {}}}) is None  # AutoGluon dict
    assert checks.verdict({"models": [{"model_id": "XGBoost_grid_1_AutoML_1"}]}) is None  # H2O list
    assert checks.verdict({"models": "0.5 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LinearL2) + 0.5 * Tuned_CatBoost"}) is None
    assert checks.verdict({"models": [{"model_id": "StackedEnsemble_AllModels_1"}, {"model_id": "GLM_1"}]})  # H2O without GBM
