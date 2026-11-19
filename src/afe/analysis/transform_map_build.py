"""Builds the transformation map tables (`python -m afe.analysis.transform_map_build`, run by `make article`).

Every cell is defined here: what the tool does, a short label for the picture, the status and the evidence. Source
references are resolved by searching the installed packages of the tool envs (`envs/<name>/.venv`, from `make env`) for
the cited line, so a missing line or env is an error rather than a stale number; run artifacts are the exp01, exp03,
exp05 and exp06 tables listed per cell. Writes results/tidy/transform_map.csv (cell text and the `<tool>_short` labels)
and transform_map_evidence.csv (does_it, status, source lines, run artifacts, note). No cell is edited by hand
afterwards: a correction goes into the definitions below.
"""
import csv
import sys
from pathlib import Path

from afe.paths import RESULTS, ROOT


def SP(e):
    (site,) = (ROOT / "envs" / e / ".venv" / "lib").glob("python3*/site-packages")
    return site


PK = {
    "agf": ("autogluon.features==1.6.3", SP("autogluon")),
    "agc": ("autogluon.common==1.6.3", SP("autogluon")),
    "agt": ("autogluon.tabular==1.6.3", SP("autogluon")),
    "lama": ("lightautoml==0.4.2", SP("lama")),
    "openfe": ("openfe==0.0.12", SP("openfe")),
    "autofeat": ("autofeat==2.1.3", SP("autofeat")),
    "mljar": ("mljar-supervised==1.3.2", SP("mljar")),
    "h2o": ("h2o==3.46.0.8", SP("h2o")),
    "skrub": ("skrub==0.11.0", SP("skrub")),
    "afe": ("afe==0.1.0", ROOT),
}
FAMILIES = ["types", "categorical_encoding", "interactions", "arithmetic", "unary", "groupby", "dates", "text", "selection"]
TOOLS = ["ag_default", "ag_tabprep", "lama", "h2o", "openfe", "autofeat", "mljar_golden", "manual_baseline", "skrub_tv"]
T = "results/tidy/"

_cache = {}


def line(pk, rel, sub, nth=1):
    f = PK[pk][1] / rel
    lines = _cache.setdefault(f, f.read_text().splitlines())
    hits = [i + 1 for i, s in enumerate(lines) if sub in s]
    if len(hits) < nth:
        sys.exit(f"NOT FOUND {pk} {rel} {sub!r}")
    return hits[nth - 1]


CELLS = {}


def cell(tool, family, does, short, what, refs, runs, status, note=""):
    assert "/Users" not in what and ";" not in what, (tool, family)
    CELLS[(family, tool)] = dict(does=does, short=short, what=what, refs=refs, runs=runs, status=status, note=note)


# ---------------------------------------------------------------- ag_default
AGP = "autogluon/features/generators/"
R_AGPIPE = ("agf", AGP + "auto_ml_pipeline.py", "generator_group = []")
R_AGGROUP_END = ("agf", AGP + "auto_ml_pipeline.py", "generators = [generator_group]")
cell("ag_default", "types", "yes", "retype, drop junk",
     "AutoMLPipelineFeatureGenerator retypes and drops, it builds no features: numeric kept, object and category become pandas category, two-valued columns become int8 bool (convert_bool=True), date-like object becomes int64 plus parts, long free text becomes n-gram columns, int-coded categories and YYYYMMDD ints stay int64. Dropped: constant ('useless'), duplicate and all-unique columns",
     [("agf", AGP + "pipeline.py", "pre_drop_useless=True,"), ("agf", AGP + "astype.py", "convert_bool: bool = True,"),
      ("agf", AGP + "drop_unique.py", "max_unique_ratio=0.99")], [T + "exp01_types.csv"], "confirmed_run",
     "pipeline output only, model-level handling not covered")
cell("ag_default", "categorical_encoding", "partial", "category dtype only",
     "no encoding: CategoryFeatureGenerator makes a pandas category, categories seen fewer than minimum_cat_count=2 times in fit become NaN, original order kept. No target, frequency or one-hot encoding in the pipeline",
     [("agf", AGP + "category.py", "minimum_cat_count: int = 2,"), ("agf", AGP + "category.py", "rank = rank[rank >= self._minimum_cat_count]")],
     [T + "exp01_types.csv"], "confirmed_run", "pipeline output only, model-level handling not covered")
for fam, what, runs in [
    ("interactions", "not done: the default generator group has identity, category, datetime, text-special, text-ngram and vision generators only", [T + "exp05_planted.csv"]),
    ("arithmetic", "not done: no arithmetic generator in the default group, planted ratio and product were found in 0 of 30 units each and no column was created", [T + "exp05_planted.csv", T + "exp01_types.csv"]),
    ("unary", "not done: no log, root, power or binning generator in the default group (BinnedFeatureGenerator exists in the package but is not used)", [T + "exp01_types.csv"]),
    ("groupby", "not done: no group-by generator in the default group (GroupByFeatureGenerator exists but is not used), planted group deviation found in 0 of 30 units", [T + "exp05_planted.csv"]),
]:
    cell("ag_default", fam, "no", "not done", what, [R_AGPIPE, R_AGGROUP_END], runs, "confirmed_run")
cell("ag_default", "dates", "yes", "timestamp + y/m/d/dow",
     "DatetimeFeatureGenerator: datetime columns and object columns where a 500-row sample is not numeric and at most 80% fail to parse as dates become a UTC timestamp as int64 plus year, month, day, dayofweek, NaT filled with the mean timestamp. YYYYMMDD ints and unix ints are not detected and stay int64",
     [("agc", "autogluon/common/features/infer_types.py", "def check_if_datetime_as_object_feature"),
      ("agc", "autogluon/common/features/infer_types.py", "if result.isnull().mean() > 0.8"),
      ("agf", AGP + "datetime.py", 'features = ["year", "month", "day", "dayofweek"]'),
      ("agf", AGP + "datetime.py", "self._fillna_map[feature] = pd.to_datetime(int(good_rows.mean())")],
     [T + "exp01_types.csv"], "confirmed_run")
cell("ag_default", "text", "yes", "stats + n-grams",
     "an object column is text if its unique ratio is above 0.01 (sample up to 5000 rows) and unique values average 3 or more words. Then TextSpecial (char_count, word_count, capital_ratio, counts and ratios of 16 symbols that occur often enough) and TextNgram (CountVectorizer min_df=30, ngram_range=(1,3), max_features=10000, uint8, one shared vocabulary): 503 columns on 5000 rows, 12 on 500 rows. Few-valued strings stay category",
     [("agc", "autogluon/common/features/infer_types.py", "if unique_ratio <= 0.01"), ("agc", "autogluon/common/features/infer_types.py", "if avg_words < 3"),
      ("agf", AGP + "text_special.py", "min_occur_ratio=0.01,"), ("agf", "autogluon/features/vectorizers.py", "CountVectorizer(min_df=30"),
      ("agf", AGP + "text_ngram.py", 'vectorizer_strategy="combined",')],
     [T + "exp01_types.csv"], "confirmed_run")
cell("ag_default", "selection", "partial", "drop const, dups, IDs",
     "structural drops only, nothing model-based: constant columns, exact duplicates (post_drop_duplicates=True) and category or object columns with more than 99% unique values (DropUnique)",
     [("agf", AGP + "pipeline.py", "post_drop_duplicates=True,"), ("agf", AGP + "drop_unique.py", "max_unique_ratio=0.99")],
     [T + "exp01_types.csv"], "confirmed_run")

# ---------------------------------------------------------------- ag_tabprep
TP = [("agt", "autogluon/tabular/models/tabprep/prep_mixin.py", 'remove_unused_features="false_recursive"'),
      ("agt", "autogluon/tabular/models/tabprep/prep_mixin.py", "passthrough_types=passthrough_types")]
cell("ag_tabprep", "types", "partial", "as ag_default, then group",
     "the _r13 group runs after AutoMLPipelineFeatureGenerator (same retyping as ag_default) inside a BulkFeatureGenerator with passthrough=True and passthrough_types invalid_raw_types category and object, so raw categoricals do not pass through (target encodings replace them, 0 category columns out). Arithmetic skips datetime-as-int, text, n-gram and text-special columns",
     TP + [("agf", AGP + "arithmetic/preprocessor.py", "S_DATETIME_AS_INT,", 2)],
     [T + "exp01_tabprep.csv", T + "exp01_prep_configs.csv"], "confirmed_run", "_r13 = GBM_PREP config of the zeroshot_2025_12_18_cpu portfolio")
cell("ag_tabprep", "categorical_encoding", "yes", "OOF target enc., alpha 10",
     "OOFTargetEncodingFeatureGenerator encodes every category column and every CatInteraction output: n_splits=5 (stratified for classification) out-of-fold mean smoothed as (mean*count + alpha*global)/(count+alpha) with alpha=10, one column per class for multiclass, keep_original=False. Out-of-fold values come only from fit_transform, transform on fit rows gives in-sample encodings. No frequency or one-hot encoding",
     [("agf", AGP + "oof_target_encoder.py", "n_splits: int = 5,"), ("agf", AGP + "oof_target_encoder.py", "alpha: float = 10.0,"),
      ("agf", AGP + "oof_target_encoder.py", "keep_original: bool = False,"),
      ("agf", AGP + "oof_target_encoder.py", "num_all = mean_all * count_all[:, None] + alpha * global_mean[None, :]")],
     [T + "exp05_te_delivery.csv", T + "exp01_tabprep.csv"], "confirmed_run")
cell("ag_tabprep", "interactions", "yes", "cat pairs, depth 3, TE",
     "CategoricalInteractionFeatureGenerator joins category columns with '_&_' up to max_order=3, at most max_new_feats=100, min_count=2, passthrough=True, and the joined columns are then target-encoded. Planted category x category table found in 30 of 30 units",
     [("agf", AGP + "cat_int.py", "max_order: int = 3,"), ("agf", AGP + "cat_int.py", "max_new_feats: int = 100,"), ("agf", AGP + "cat_int.py", "min_count: int = 2,")],
     [T + "exp05_planted.csv", T + "exp01_prep_configs.csv"], "confirmed_run")
cell("ag_tabprep", "arithmetic", "yes", "+ - * / random pick",
     "ArithmeticFeatureGenerator: + - * / (and the inverse ratio) up to order max_order=3 over numeric columns with at least min_cardinality=3 values (cat_as_num=False), max_base_feats=150, max_new_feats=2000, selection_method='random' with no correlation or model filter (corr_threshold is used only by 'spearman'), row subsample 100000. 894 new columns from 6 base columns on the hard_columns frame. Planted ratio and product found in 30 of 30 units each",
     [("agf", AGP + "arithmetic/preprocessor.py", 'selection_method: Literal["spearman", "random"] = "random",'),
      ("agf", AGP + "arithmetic/preprocessor.py", "max_order: int = 3,"), ("agf", AGP + "arithmetic/preprocessor.py", "max_new_feats: int = 2000,"),
      ("agf", AGP + "arithmetic/preprocessor.py", "min_cardinality: int = 3,"),
      ("agf", AGP + "arithmetic/combinations.py", 'interaction_types=["/", "*", "-", "+"],')],
     [T + "exp01_tabprep.csv", T + "exp05_planted.csv"], "confirmed_run")
cell("ag_tabprep", "unary", "no", "not done",
     "not done: the _r13 group has Arithmetic, CatInteraction and OOF-TE only. Binned, CatAsNum and Frequency generators exist in the package but none of the 19 inspected GBM_PREP configs (2 portfolios) uses them",
     [("agt", "autogluon/tabular/models/tabprep/prep_mixin.py", "def get_preprocessor_tabprep")], [T + "exp01_prep_configs.csv"], "confirmed_run")
cell("ag_tabprep", "groupby", "no", "not used",
     "not done: GroupByFeatureGenerator exists (mean and pct_rank per category, ratio to the mean, max_features=500) but groupby is False in all 19 inspected GBM_PREP configs. Planted group deviation found in 0 of 30 units",
     [("agf", AGP + "groupby.py", "max_features=500,")], [T + "exp01_prep_configs.csv", T + "exp05_planted.csv"], "confirmed_run")
cell("ag_tabprep", "dates", "partial", "as ag_default, no arith.",
     "date parts come from the AutoML pipeline exactly as in ag_default, the group adds nothing for dates: date columns are excluded from Arithmetic (datetime-as-int is an invalid special type), 6 arithmetic base columns on hard_columns and none from the date column",
     [("agf", AGP + "arithmetic/preprocessor.py", "S_DATETIME_AS_INT,", 2)], [T + "exp01_tabprep.csv"], "confirmed_run")
cell("ag_tabprep", "text", "partial", "as ag_default, no arith.",
     "text columns come from the AutoML pipeline exactly as in ag_default (special stats and n-grams), the group adds nothing for text: text, n-gram and text-special columns are excluded from Arithmetic (hard_columns has 525 input columns, 6 are arithmetic bases)",
     [("agf", AGP + "arithmetic/preprocessor.py", "S_TEXT_NGRAM,", 2)], [T + "exp01_tabprep.csv"], "confirmed_run")
cell("ag_tabprep", "selection", "partial", "random pick, dedupe",
     "no importance-based selection: Arithmetic picks at random under max_new_feats, the wrapper sets remove_unused_features='false_recursive' and post_drop_duplicates=True. How many columns the wrapper removes was not measured",
     TP + [("agf", AGP + "arithmetic/preprocessor.py", 'selection_method: Literal["spearman", "random"] = "random",')], [], "confirmed_source")

# ---------------------------------------------------------------- lama
LP = "lightautoml/"
cell("lama", "types", "yes", "Reader guesses roles",
     "Reader assigns Numeric, Category, Datetime or Drop per column: castable to float is Numeric, parsable as date is Datetime, else Category. advanced_roles then scores raw, binned and encoded variants by Gini plus rules, flips numeric to category (int-coded city_code, adult age and fnlwgt) and picks the category encoding (auto, freq, ordinal). Columns with NaN or constant rate of 0.999 or more are dropped, ID-like columns stay as ordinal category",
     [("lama", LP + "reader/base.py", "def _guess_role"), ("lama", LP + "reader/base.py", "return CategoryRole(object)"),
      ("lama", LP + "reader/base.py", "if self.advanced_roles:"), ("lama", LP + "reader/guess_roles.py", "def rule_based_roles_guess")],
     [T + "exp03_roles.csv", T + "exp03_role_stats.csv", T + "exp03_config.yaml"], "confirmed_run", "default TabularAutoML, pandas<2 env")
cell("lama", "categorical_encoding", "yes", "OOF TE, ordinal, freq",
     "GBM pipeline (lgb, cb): categories with 10 or fewer levels (auto_unique_co) get ordinal codes, more levels or an oof role get out-of-fold target encoding over the reader's 5 folds (smoothing alpha picked per column from 0.5 to 1000 by OOF score), freq role gets frequency encoding. Multiclass target encoding only with 3 or fewer classes (multiclass_te_co). linear_l2 uses label codes plus target encoding then log-odds then scaling. Observed names: ord__x, oof__le__x, freq__x",
     [("lama", LP + "pipelines/features/lgb_pipeline.py", "auto_unique_co: int = 10,"),
      ("lama", LP + "pipelines/features/lgb_pipeline.py", "te_part = SequentialTransformer([te_part, target_encoder()])"),
      ("lama", LP + "pipelines/features/base.py", "def get_target_encoder"),
      ("lama", LP + "transformers/categorical.py", "alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0)")],
     [T + "exp03_fit_models.csv", T + "exp03_config.yaml", T + "exp05_te_delivery.csv"], "confirmed_run")
cell("lama", "interactions", "yes", "top-4 cats, depth 3, TE",
     "categorical intersections of the top top_intersections=4 categories (selector importance, else cardinality) up to max_intersection_depth=3, each target-encoded out of fold (oof__inter__(a__b__c)). Planted category x category table found in 30 of 30 units",
     [("lama", LP + "pipelines/features/base.py", "def get_categorical_intersections"), ("lama", LP + "pipelines/features/base.py", "def get_top_categories"),
      ("lama", LP + "pipelines/features/lgb_pipeline.py", "intersections = self.get_categorical_intersections(train)")],
     [T + "exp03_fit_models.csv", T + "exp05_planted.csv", T + "exp03_config.yaml"], "confirmed_run")
cell("lama", "arithmetic", "no", "not done",
     "not done: numeric columns pass as they are in the GBM pipeline (get_numeric_data) and are only filled, binned or scaled in linear_l2. Planted ratio and product found in 0 of 30 units each (1 column created per unit)",
     [("lama", LP + "pipelines/features/lgb_pipeline.py", "transformer_list.append(self.get_numeric_data(train))"),
      ("lama", LP + "pipelines/features/base.py", "def get_numeric_data")], [T + "exp05_planted.csv", T + "exp03_fit_models.csv"], "confirmed_run")
cell("lama", "unary", "partial", "linear only: bins, scale",
     "GBM pipeline: none. linear_l2 only: NaN flags, median fill, standard scaling, quantile binning of numeric columns whose role has discretization (qntl__), log-odds of target encodings. No log, root or power transform among the numeric transformers",
     [("lama", LP + "transformers/numeric.py", "class QuantileBinning"), ("lama", LP + "transformers/numeric.py", "class LogOdds"),
      ("lama", LP + "pipelines/features/linear_pipeline.py", "sparse_list.append(self.get_binned_data(train))")],
     [T + "exp03_dates_text.csv", T + "exp03_fit_models.csv"], "confirmed_run")
cell("lama", "groupby", "no", "off by default",
     "off by default: the group-by transformer exists (delta_median, delta_mean, min, max, std, mode, is_mode on the top 3 categorical x top 3 numeric columns by cardinality) but use_groupby=False in the default TabularAutoML config. Planted group deviation found in 0 of 30 units",
     [("lama", LP + "pipelines/features/lgb_pipeline.py", "if self.use_groupby:"), ("lama", LP + "pipelines/features/base.py", "def get_groupby"),
      ("lama", LP + "transformers/groupby.py", "class GroupByTransformer")],
     [T + "exp03_config.yaml", T + "exp05_planted.csv"], "confirmed_run")
cell("lama", "dates", "yes", "year, month, weekday",
     "ISO strings and datetime64 columns get a Datetime role with seasonality y, m, wd: the GBM pipeline outputs season_y, season_m, season_wd as numbers (linear_l2 as label-encoded categories) and no raw timestamp. Difference to a base date (basediff) needs a manual base_date role and holidays a manual country role, neither is set by default. YYYYMMDD ints are read as Numeric. The selector sees dates as day counts relative to the first row",
     [("lama", LP + "dataset/roles.py", 'seasonality: Optional[Sequence[str]] = ("y", "m", "wd"),'),
      ("lama", LP + "pipelines/features/lgb_pipeline.py", "transformer_list.append(self.get_datetime_seasons(train, NumericRole(np.float32)))"),
      ("lama", LP + "pipelines/features/base.py", "def get_datetime_diffs"), ("lama", LP + "transformers/datetime.py", "new_arr = ((data - data.iloc[0])")],
     [T + "exp03_dates_text.csv"], "confirmed_run")
cell("lama", "text", "no", "none (needs extras)",
     "not done by TabularAutoML ('No text support'): no text role is guessed, a free-text column becomes an ordinal Category and is encoded as one huge category (the GBM selector dropped it in the probe), a manual text role produced no features in either model. TF-IDF (linear), embeddings (GBM) and BERT (NN) are configured in TabularNLPAutoML, which needs the nlp extras (transformers, nltk, fasttext) that are not installed in envs/lama: importing it raises PackageNotFoundError",
     [("lama", LP + "automl/presets/tabular_presets.py", "- No text support"), ("lama", LP + "automl/presets/text_presets.py", '__validate_extra_deps("nlp", error=True)'),
      ("lama", LP + "automl/presets/text_config.yml", 'text_features: "embed"')],
     [T + "exp03_dates_text.csv"], "confirmed_run", "extras path not run, only its import failure")
cell("lama", "selection", "yes", "LightGBM gain > 0",
     "selection mode 1: ImportanceCutoffSelector on LGBSimpleFeatures with LightGBM gain importance, fitted on the first fold as holdout, drops input features with importance at or below cutoff 0 before the pipelines are built. Dropped a free-text column in the probe",
     [("lama", LP + "pipelines/selection/importance_based.py", "class ImportanceCutoffSelector"),
      ("lama", LP + "pipelines/selection/importance_based.py", "selected = self.mapped_importances.index.values[self.mapped_importances.values > self.cutoff]")],
     [T + "exp03_config.yaml", T + "exp03_dates_text.csv", T + "exp03_fit_models.csv"], "confirmed_run")

# ---------------------------------------------------------------- h2o
H2R = [T + "exp06_h2o_probe.csv"]
cell("h2o", "types", "yes", "real, enum, time, string",
     "pandas frame is written to CSV and parsed by the server, which guessed real (float), enum (few-valued strings), time (datetime64) and string (nearly unique strings). String columns are in no model's inputs, no other retyping. XGBoost has no backend in this env, so the AutoML run has no XGBoost model",
     [("h2o", "h2o/frame.py", "def _upload_python_object")], H2R, "confirmed_run", "default and target_encoding runs give the same types")
cell("h2o", "categorical_encoding", "yes", "Enum (trees), TE opt-in",
     "default AutoML: actual categorical_encoding is Enum for GBM, DRF and XRT and OneHotInternal for DeepLearning, GLM and StackedEnsemble expose none. With preprocessing=['target_encoding']: columns with 25 or more levels (24 stayed raw, 25 encoded) become <col>_te for GBM, DRF and XRT only (DL and GLM keep the raw factor), keep_original=False, KFold on AutoML's fold column, blending=True, inflection_point=5, smoothing=10, noise=0 (library defaults: blending False, 10, 20, noise 0.01, no leakage handling)",
     [("h2o", "h2o/automl/_estimator.py", "preprocessing=None,"), ("h2o", "h2o/estimators/targetencoder.py", "blending=False,"),
      ("h2o", "h2o/estimators/targetencoder.py", 'data_leakage_handling="none",')], H2R, "confirmed_run", "XGBoost not available here, so its encoding is not covered")
for fam, what in [
    ("interactions", "not done: no interaction column in any model's inputs in either run"),
    ("arithmetic", "not done: no derived numeric column in any model's inputs in either run"),
    ("unary", "not done: no derived column in any model's inputs, scaling inside DeepLearning and GLM was not probed"),
    ("groupby", "not done: no group statistic column in any model's inputs in either run"),
]:
    cell("h2o", fam, "no", "not done", what, [("h2o", "h2o/automl/_estimator.py", "preprocessing=None,")], H2R, "confirmed_run", "inputs read from each model's output names")
cell("h2o", "dates", "partial", "one time column",
     "datetime64 is uploaded as type time (milliseconds since the Unix epoch per the client docs) and reaches every model as that one column: no year, month, day or weekday parts and no differences",
     [("h2o", "h2o/frame.py", "Long msec since the Unix Epoch")], H2R, "confirmed_run")
cell("h2o", "text", "no", "unique strings dropped",
     "not done: a string column with nearly unique values (type string) is absent from every model's inputs, a string column with 5 distinct values is parsed as enum and handled as a categorical (Enum in trees, OneHotInternal in DeepLearning). No tokenization, n-grams or embeddings",
     [("h2o", "h2o/frame.py", "def _upload_python_object")], H2R, "confirmed_run")
cell("h2o", "selection", "no", "none",
     "not done: every base model (GBM, DRF, XRT, DeepLearning, GLM) receives all non-string input columns, with or without target encoding",
     [("h2o", "h2o/automl/_estimator.py", "preprocessing=None,")], H2R, "confirmed_run")

# ---------------------------------------------------------------- openfe
OF = "openfe/"
OFN = [("openfe", OF + "FeatureGenerator.py", "all_operators = [")]
cell("openfe", "types", "partial", "numeric or categorical",
     "non-number dtypes (object, category, datetime64, bool) are categorical (select_dtypes(exclude=np.number)), numeric columns with 100 or fewer distinct values are 'ordinal' and enter both the numeric and the categorical candidate lists, no NaN imputation. The repo wrapper casts bool to int8, datetime to int64 ns and other non-numeric to str",
     [("openfe", OF + "openfe.py", "return list(self.data.select_dtypes(exclude=np.number))"), ("openfe", OF + "openfe.py", "elif self.data[feature].nunique() <= 100:"),
      ("afe", "src/afe/generators/openfe_gen.py", "def prep")], [], "confirmed_source")
cell("openfe", "categorical_encoding", "partial", "freq, pair codes",
     "no target encoding: freq (value counts) of every column, Combine (a pair as a new category) and CombineThenFreq. OpenFE's own scoring LightGBM sees categories as integer codes. The repo's inductive reimplementation of freq, Combine and CombineThenFreq matches OpenFE's values (max abs difference 0)",
     [("openfe", OF + "FeatureGenerator.py", "all_operators = ["), ("openfe", OF + "FeatureGenerator.py", 'elif self.name == "freq":'),
      ("openfe", OF + "openfe.py", "data_new[f] = data_new[f].cat.codes")], [T + "exp05_openfe_ops.csv"], "confirmed_run")
cell("openfe", "interactions", "yes", "cat pairs (Combine)",
     "Combine joins two categorical or ordinal columns into a new category, CombineThenFreq is its frequency. First order only (get_candidate_features order=1): pairs, no depth 3. Planted category x category table found in 20 of 30 units",
     [("openfe", OF + "openfe.py", "def get_candidate_features(numerical_features=None, categorical_features=None, ordinal_features=None, order=1):"),
      ("openfe", OF + "FeatureGenerator.py", 'cat_cat_operators = ["Combine"')], [T + "exp05_openfe_ops.csv", T + "exp05_planted.csv"], "confirmed_run")
cell("openfe", "arithmetic", "yes", "+ - * / min max",
     "min, max, +, -, *, / over pairs of numeric columns, each unordered pair once also for - and / (symmetry_operators), so one direction of a/b only. Planted ratio and product found in 30 of 30 units each",
     [("openfe", OF + "FeatureGenerator.py", 'num_num_operators = ["min"'), ("openfe", OF + "FeatureGenerator.py", "symmetry_operators = ["),
      ("openfe", OF + "openfe.py", "k = i if op in symmetry_operators else 0")], [T + "exp05_openfe_ops.csv", T + "exp05_planted.csv"], "confirmed_run")
cell("openfe", "unary", "yes", "abs log sqrt sigmoid",
     "abs, log (log of |x|, 0 becomes NaN), sqrt of |x|, square, sigmoid, round (floor) and residual (x - floor x) on every numeric column",
     [("openfe", OF + "FeatureGenerator.py", "num_operators = ["), ("openfe", OF + "FeatureGenerator.py", 'new_data = np.log(np.abs(d.replace(0, np.nan)))')],
     [T + "exp05_openfe_ops.csv"], "confirmed_run")
cell("openfe", "groupby", "yes", "mean std rank by cat",
     "GroupByThenMin, Max, Mean, Median, Std and Rank (numeric column by a categorical or ordinal key) and GroupByThenNUnique (categorical by categorical). The statistics are column-wide: OpenFE's own transform computes them on train and test together, the repo's wrapper refits them on fit rows. Planted group deviation found in 30 of 30 units",
     [("openfe", OF + "FeatureGenerator.py", 'cat_num_operators = ["GroupByThenMin"'), ("openfe", OF + "FeatureGenerator.py", "elif self.name == 'GroupByThenRank':"),
      ("openfe", OF + "utils.py", "def transform(X_train, X_test, new_features_list, n_jobs, name=\"\"):")],
     [T + "exp05_openfe_ops.csv", T + "exp05_transductive.csv", T + "exp05_planted.csv"], "confirmed_run")
cell("openfe", "dates", "no", "not done",
     "not done: no calendar features. A datetime64 column is non-numeric, so the library treats it as a categorical (freq, Combine and group key on exact timestamps), the repo wrapper casts it to int64 ns so numeric operators apply to the raw timestamp",
     [("openfe", OF + "openfe.py", "return list(self.data.select_dtypes(exclude=np.number))"), ("afe", "src/afe/generators/openfe_gen.py", "elif pd.api.types.is_datetime64_any_dtype(s):")],
     [], "confirmed_source")
cell("openfe", "text", "no", "not done",
     "not done: an object column is a categorical (freq, Combine and group key on the whole string), the repo wrapper casts it to str. No tokenization or vectorization",
     [("openfe", OF + "openfe.py", "return list(self.data.select_dtypes(exclude=np.number))"), ("afe", "src/afe/generators/openfe_gen.py", "elif not pd.api.types.is_numeric_dtype(s):")],
     [], "confirmed_source")
cell("openfe", "selection", "yes", "halving + LightGBM gain",
     "stage 1: successive feature-wise halving over n_data_blocks=8 with a LightGBM 'predictive' score keeps candidates with score above 0 (min_candidate_features=2000 floor, top 100 if none), equal-score neighbours are removed (delete_same). Stage 2: LightGBM (1000 trees, 16 leaves, early stopping 50) gain importance ranks the survivors. fit returns the ranked list with no cut-off (the repo keeps gain above 0, cap 2000)",
     [("openfe", OF + "openfe.py", "n_data_blocks=8,"), ("openfe", OF + "openfe.py", "min_candidate_features=2000,"),
      ("openfe", OF + "openfe.py", "return_results = [item[0] for item in candidate_features_scores if item[1] > 0]"),
      ("openfe", OF + "openfe.py", 'params = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 16,'),
      ("openfe", OF + "openfe.py", "def delete_same")], [], "confirmed_source")

# ---------------------------------------------------------------- autofeat
AF = "autofeat/"
cell("autofeat", "types", "partial", "numeric; one-hot cats",
     "numeric inputs: non-numeric columns must be listed in categorical_cols (one-hot, then numeric 0/1 columns that are also engineered), the docstring asks for finite values. The repo wrapper engineers numeric columns only, fills NaN with fit medians, one-hot encodes categoricals with 50 or fewer levels itself and leaves wider ones out",
     [("autofeat", AF + "autofeat.py", "e = OneHotEncoder(sparse_output=False, categories=\"auto\")"),
      ("autofeat", AF + "autofeat.py", "The data should only contain finite values (no NaNs etc.)"),
      ("afe", "src/afe/generators/autofeat_gen.py", "MAX_LEVELS = 50")], [], "confirmed_source")
cell("autofeat", "categorical_encoding", "partial", "one-hot only",
     "one-hot only: OneHotEncoder(categories='auto') for columns in categorical_cols, new columns cat_<col>_<level>, refitted on every call (so the levels of a transform batch can differ). No target or frequency encoding",
     [("autofeat", AF + "autofeat.py", "e = OneHotEncoder(sparse_output=False, categories=\"auto\")"),
      ("autofeat", AF + "autofeat.py", "ohe = e.fit_transform(df[c].to_numpy()[:, None])")], [], "confirmed_source")
cell("autofeat", "interactions", "no", "not done",
     "not done: no categorical operator, only numeric combinations (one-hot dummies are numeric inputs and enter the arithmetic like any column). Planted category x category table found in 0 of 30 units",
     [("autofeat", AF + "feateng.py", '"x+y": lambda x, y: x + y,')], [T + "exp05_planted.csv"], "confirmed_run")
cell("autofeat", "arithmetic", "yes", "+ - * pairs, 2 steps",
     "feateng_steps=2: each step applies the unary transformations, then pairwise x+y, x*y, x-y, y-x over the feature pool (n_cols_generated bounds the matrix, max_gb subsamples rows above it), ratios arise as x * (1/y). Planted ratio and product found in 30 of 30 units each with about 4 and 6 columns kept",
     [("autofeat", AF + "autofeat.py", "feateng_steps: int = 2,"), ("autofeat", AF + "feateng.py", '"x+y": lambda x, y: x + y,'), ("autofeat", AF + "feateng.py", '"y-x": lambda x, y: y - x,')],
     [T + "exp05_planted.csv"], "confirmed_run")
cell("autofeat", "unary", "yes", "1/x exp log abs powers",
     "transformations=('1/', 'exp', 'log', 'abs', 'sqrt', '^2', '^3') by default (optional 1+, 1-, sin, cos, exp-, 2^), each applied only where defined (domain checks such as x != 0 for 1/x)",
     [("autofeat", AF + "autofeat.py", 'transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),'),
      ("autofeat", AF + "feateng.py", '"1/": lambda x: np.all(x != 0),')], [], "confirmed_source")
cell("autofeat", "groupby", "no", "not done",
     "not done: no group or aggregation operator in the package (only unary transformations and pairwise arithmetic). Planted group deviation found in 0 of 30 units",
     [("autofeat", AF + "feateng.py", "def get_feature_combinations")], [T + "exp05_planted.csv"], "confirmed_run")
cell("autofeat", "dates", "no", "not done",
     "not done: no date code in the package (checked by grep) and input must be numeric or listed in categorical_cols. The repo wrapper treats a datetime64 column as a non-numeric column (one-hot up to 50 levels, else left out)",
     [("autofeat", AF + "autofeat.py", "X, target = check_X_y(X, y, y_numeric=self.problem_type == \"regression\", dtype=None)"),
      ("afe", "src/afe/generators/autofeat_gen.py", "self.left_out = [c for c in others if nunique[c] > MAX_LEVELS]")], [], "confirmed_source")
cell("autofeat", "text", "no", "not done",
     "not done: no text code in the package (checked by grep). The repo wrapper treats a string column as a categorical (one-hot up to 50 levels, else left out)",
     [("autofeat", AF + "autofeat.py", "e = OneHotEncoder(sparse_output=False, categories=\"auto\")"),
      ("afe", "src/afe/generators/autofeat_gen.py", "self.left_out = [c for c in others if nunique[c] > MAX_LEVELS]")], [], "confirmed_source")
cell("autofeat", "selection", "yes", "L1 + noise features",
     "featsel_runs=5 runs on random data fractions: LassoLarsCV (regression) or L1 LogisticRegressionCV (classification) with added noise features, keep features whose coefficient beats the noise, union over runs, drop features correlated 0.9 or more with a kept one, final noise filter. Generation drops generated features correlated above 0.95 with an original",
     [("autofeat", AF + "autofeat.py", "featsel_runs: int = 5,"), ("autofeat", AF + "featsel.py", "model = lm.LassoLarsCV(cv=5, eps=1e-8)"),
      ("autofeat", AF + "featsel.py", 'model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")'),
      ("autofeat", AF + "featsel.py", "if np.max(np.abs(correlations[c].ravel()[:i])) < 0.9:"), ("autofeat", AF + "feateng.py", "if corr < 0.95:")],
     [], "confirmed_source")

# ---------------------------------------------------------------- mljar_golden
MJ = "supervised/preprocessing/goldenfeatures_transformer.py"
MPRE = ("mljar", "supervised/preprocessing/preprocessing.py", 'self._golden_features.fit(X_train[numeric_cols], y_train)')
cell("mljar_golden", "types", "partial", "numeric only",
     "numeric columns only: MLJAR's own caller passes X_train.select_dtypes(include='number') and the transformer fits on whatever it gets (the repo wrapper fills NaN with fit medians). Other columns are untouched",
     [("mljar", "supervised/preprocessing/preprocessing.py", 'numeric_cols = X_train.select_dtypes(include="number").columns.tolist()'), MPRE,
      ("afe", "src/afe/generators/mljar_golden.py", "self.cols = _numeric(X_fit)")], [], "confirmed_source")
for fam, what in [
    ("categorical_encoding", "not done by GoldenFeaturesTransformer: it only sees numeric columns (MLJAR AutoML's own categorical preprocessing is not part of this arm)"),
    ("interactions", "not done: only numeric pairs, no categorical operator"),
    ("unary", "not done: the five operations are all pairwise (diff, two ratios, sum, multiply)"),
    ("groupby", "not done: no group or aggregation operation"),
    ("dates", "not done by GoldenFeaturesTransformer: datetime columns are not numeric and never reach it (MLJAR AutoML has its own DateTimeTransformer, not part of this arm)"),
    ("text", "not done by GoldenFeaturesTransformer: text columns never reach it (MLJAR AutoML has its own TextTransformer, not part of this arm)"),
]:
    cell("mljar_golden", fam, "no", "not done", what, [MPRE], [T + "exp05_planted.csv"] if fam in ("interactions", "groupby") else [], "confirmed_source")
CELLS[("interactions", "mljar_golden")]["status"] = "confirmed_run"
CELLS[("groupby", "mljar_golden")]["status"] = "confirmed_run"
CELLS[("interactions", "mljar_golden")]["what"] += ". Planted category x category table found in 0 of 30 units"
CELLS[("groupby", "mljar_golden")]["what"] += ". Planted group deviation found in 0 of 30 units"
cell("mljar_golden", "arithmetic", "yes", "diff, ratio, sum, product",
     "all unordered pairs of numeric columns with diff (a-b), ratio a/b and b/a (0 where the denominator is 0), sum and multiply, each scored by a depth-3 decision tree on a subsample (up to 10000 rows, 2500 train, the rest test, a quarter below). Above 250000 pairs a random sample of pairs (unseeded np.random). Planted ratio and product found in 30 of 30 units each",
     [("mljar", MJ, "combinations = itertools.combinations(X.columns, r=2)"), ("mljar", MJ, "clf = DecisionTreeClassifier(max_depth=3)"),
      ("mljar", MJ, "MAX_SIZE = 10000"), ("mljar", MJ, "X[new_col] = np.divide("), ("mljar", MJ, "if len(items) > 250000:")],
     [T + "exp05_planted.csv"], "confirmed_run")
cell("mljar_golden", "selection", "yes", "top by tree score",
     "keeps the pairs with the lowest tree log loss or error: new_cols_cnt = min(100, max(10, 10% of columns)) (10 on every dataset here with 100 or fewer columns), or features_count when given. No comparison with the original columns",
     [("mljar", MJ, "new_cols_cnt = np.min([100, np.max([10, int(0.1 * X.shape[1])])])"), ("mljar", MJ, 'df.sort_values(by="score", inplace=True)')],
     [T + "exp05_planted.csv"], "confirmed_run", "created = 10 columns in every planted unit")

# ---------------------------------------------------------------- manual_baseline
MB = "src/generators"
MBP = "src/afe/generators/manual_baseline.py"
cell("manual_baseline", "types", "yes", "numeric vs the rest",
     "numeric = numeric non-bool columns, categorical = every other non-bool column (object, category and datetime64). The generator gets the raw frame",
     [("afe", MBP, "def _is_num(s: pd.Series) -> bool:"), ("afe", MBP, "cat = [c for c in X_fit.columns if not _is_num(X_fit[c])")], [], "confirmed_source")
cell("manual_baseline", "categorical_encoding", "yes", "frequency",
     "frequency encoding of every categorical column: share of fit rows, NaN is its own level, unseen level gets 0",
     [("afe", MBP, "self.freq = {c: _key(X_fit[c]).value_counts(normalize=True) for c in cat}"), ("afe", MBP, 'out = {f"freq__{c}": _key(X[c]).map(f).fillna(0.0).astype(float)')],
     [], "confirmed_source")
cell("manual_baseline", "interactions", "no", "not done",
     "not done: no categorical combination. Planted category x category table found in 0 of 30 units",
     [("afe", MBP, 'out = {f"freq__{c}": _key(X[c]).map(f).fillna(0.0).astype(float)')], [T + "exp05_planted.csv"], "confirmed_run")
cell("manual_baseline", "arithmetic", "yes", "ratio, product of top 5",
     "top 5 numeric columns by mutual information with y (random_state=seed, NaN filled with medians), for each pair a/b, b/a (denominator below 1e-12 in absolute value gives NaN) and a*b: 30 columns. Planted ratio and product found in 30 of 30 units each",
     [("afe", MBP, "TOP = 5"), ("afe", MBP, "EPS = 1e-12"), ("afe", MBP, 'out[f"{a}__div__{b}"] = xa / xb.where(xb.abs() >= EPS)'), ("afe", MBP, 'out[f"{a}__mul__{b}"] = xa * xb')],
     [T + "exp05_planted.csv"], "confirmed_run")
cell("manual_baseline", "unary", "no", "not done", "not done: no single-column transform",
     [("afe", MBP, "def transform(self, X) -> pd.DataFrame:")], [], "confirmed_source")
cell("manual_baseline", "groupby", "yes", "group mean, diff",
     "key = the categorical column with the highest mutual information with y among those with at most 50% unique values (none: decile bins of the top numeric column), then group mean and x minus group mean of each top-5 numeric column, unseen key gets the global mean. Planted group deviation found in 14 of 30 units",
     [("afe", MBP, "NEAR_UNIQUE = 0.5"), ("afe", MBP, 'out[f"{c}__minus_gmean__{self.key_col}"] = X[c].astype(float) - gm'),
      ("afe", MBP, "self.bins = np.unique(X_fit[self.key_col].quantile(np.linspace(0, 1, 11)).to_numpy())")],
     [T + "exp05_planted.csv"], "confirmed_run", "one key column only")
cell("manual_baseline", "dates", "no", "not done",
     "not done: a datetime64 column is non-numeric, so it is frequency-encoded as a categorical (exact timestamps) and may serve as the group key",
     [("afe", MBP, "def _is_num(s: pd.Series) -> bool:"), ("afe", MBP, "cat = [c for c in X_fit.columns if not _is_num(X_fit[c])")], [], "confirmed_source")
cell("manual_baseline", "text", "no", "not done",
     "not done: a string column is a categorical, frequency-encoded as a whole string",
     [("afe", MBP, "cat = [c for c in X_fit.columns if not _is_num(X_fit[c])")], [], "confirmed_source")
cell("manual_baseline", "selection", "partial", "top-5 inputs by MI",
     "selects inputs, not outputs: the top 5 numeric columns by mutual information feed the arithmetic and group features, no filtering of the generated columns",
     [("afe", MBP, "self.top = num_mi.sort_values(ascending=False, kind=\"stable\").index[:TOP].tolist()")], [], "confirmed_source")


# ---------------------------------------------------------------- skrub_tv
TVP = "skrub/_table_vectorizer.py"
TV_DEF = ("skrub", TVP, "HIGH_CARDINALITY_TRANSFORMER = StringEncoder(n_components=30)")
TV_RUN = [T + "exp08_skrub_types.csv", T + "exp08_skrub_defaults.yaml"]
cell("skrub_tv", "types", "yes", "type, then cardinality",
     "TableVectorizer sorts each column into numeric, datetime, low-cardinality or high-cardinality string, cleans null strings, converts strings that hold only numbers to float32 (num_as_str and the mixed column with 'n/a' become numeric, 'n/a' becomes NaN), parses date strings, and casts every output to float32. Nothing is dropped except all-null columns: constant, duplicate and all-unique columns stay (probe: 0 of 114 columns dropped)",
     [("skrub", TVP, "HIGH_CARDINALITY_TRANSFORMER = StringEncoder(n_components=30)"), ("skrub", TVP, "drop_null_fraction=1.0,"),
      ("skrub", TVP, "drop_if_constant=False,")], TV_RUN, "confirmed_run", "pipeline output only; the output has no category columns")
cell("skrub_tv", "categorical_encoding", "yes", "one-hot below 40, else SVD",
     "string columns with fewer than 40 unique values (cardinality_threshold=40) are one-hot encoded, dropping one column when binary, unseen levels give all zeros, and columns with 40 or more unique values go to StringEncoder: char 3-4-gram TF-IDF reduced by SVD to 30 columns per input column. No target or frequency encoding, no y used. Output is all float32, so CatBoost sees no categorical column",
     [("skrub", TVP, "cardinality_threshold=40,"), ("skrub", TVP, "LOW_CARDINALITY_TRANSFORMER = OneHotEncoder("),
      ("skrub", TVP, 'drop="if_binary",'), ("skrub", TVP, 'handle_unknown="ignore",'), TV_DEF], TV_RUN, "confirmed_run")
for fam, what in [
    ("interactions", "not done: TableVectorizer encodes each column independently, a different transformer instance per column"),
    ("arithmetic", "not done: no numeric combination, numeric columns pass through (PassThrough) as float32"),
    ("unary", "not done: no log, root, power or binning of numeric columns"),
    ("groupby", "not done: no group-by feature, columns are processed independently"),
]:
    cell("skrub_tv", fam, "no", "not done", what, [("skrub", TVP, "NUMERIC_TRANSFORMER = PassThrough()")] if fam in ("arithmetic", "unary") else [TV_DEF], TV_RUN, "confirmed_run")
cell("skrub_tv", "dates", "yes", "year, month, day, seconds",
     "datetime columns and date strings that parse as dates go to DatetimeEncoder (resolution='hour', add_weekday=False, add_total_seconds=True): year, month, day and total seconds since the epoch, finer parts only when the column has times (probe: date_str gives 4 columns). YYYYMMDD ints and unix ints are numeric and stay as they are",
     [("skrub", "skrub/_datetime_encoder.py", 'resolution="hour",'), ("skrub", "skrub/_datetime_encoder.py", "add_total_seconds=True,")],
     TV_RUN, "confirmed_run")
cell("skrub_tv", "text", "partial", "SVD of char n-grams",
     "no text-specific features: a string column with 40 or more unique values, free text included, goes to StringEncoder (char_wb 3-4-gram TF-IDF, SVD to 30 columns), the same path as an ID column (uid and text_long give 30 columns each in the probe). Short-text columns with fewer than 40 values are one-hot encoded as categories",
     [TV_DEF, ("skrub", "skrub/_string_encoder.py", 'analyzer="char_wb",'), ("skrub", "skrub/_string_encoder.py", "ngram_range=(3, 4),")],
     TV_RUN, "confirmed_run")
cell("skrub_tv", "selection", "partial", "drops all-null only",
     "DropUninformative with drop_null_fraction=1.0 drops columns that are entirely null, while constant columns (drop_if_constant=False), duplicates and unique columns are kept. No model-based selection",
     [("skrub", TVP, "drop_null_fraction=1.0,"), ("skrub", TVP, "drop_if_constant=False,")], TV_RUN, "confirmed_run")


SHORT = {
 "ag_default": ["retype, drop junk", "category dtype only", "not done", "not done", "not done", "not done", "int stamp + 4 parts", "special stats + n-grams", "drops const, dups, IDs"],
 "ag_tabprep": ["AG default first", "OOF target enc., alpha 10", "cat pairs, depth 3, TE", "+ - * / random pick", "not done", "exists, unused", "as AG default", "as AG default", "random pick, dedupe"],
 "lama": ["Reader guesses roles", "OOF TE, ordinal, freq", "top-4 cats, depth 3, TE", "not done", "linear: bins, scale", "off by default", "year, month, weekday", "none, needs extras", "GBM gain > 0"],
 "h2o": ["real, enum, time, string", "Enum trees, TE opt-in", "not done", "not done", "not done", "not done", "one time column", "unique strings dropped", "not done"],
 "openfe": ["numeric or category", "freq, pair codes", "cat pairs: Combine", "+ - * / min max", "abs log sqrt sigmoid", "mean std rank by cat", "not done", "not done", "halving, GBM gain"],
 "autofeat": ["numeric, one-hot cats", "one-hot only", "not done", "+ - * pairs, 2 steps", "1/x exp log powers", "not done", "not done", "not done", "L1 with noise cols"],
 "mljar_golden": ["numeric only", "not done", "not done", "diff, ratio, sum, product", "not done", "not done", "not done", "not done", "top by tree score"],
 "skrub_tv": ["type, then cardinality", "one-hot below 40, else SVD", "not done", "not done", "not done", "not done", "year, month, day, seconds", "SVD of char n-grams", "drops all-null only"],
 "manual_baseline": ["numeric vs rest", "freq share", "not done", "ratio, product, top 5", "not done", "group mean, diff", "not done", "not done", "top 5 by MI"],
}
for _t, _l in SHORT.items():
    for _f, _s in zip(FAMILIES, _l):
        CELLS[(_f, _t)]["short"] = _s


def render():
    rows, ev = [], []
    for fam in FAMILIES:
        row = {"family": fam}
        short = {}
        for tool in TOOLS:
            c = CELLS[(fam, tool)]
            res = [(PK[k][0], p, line(k, p, r[0], *r[1:])) for k, p, *r in c["refs"]]
            parts = [c["what"]]
            if res:
                parts.append("ref: " + ", ".join(f"{pk} {p}:{ln}" for pk, p, ln in res))
            if c["runs"]:
                parts.append("run: " + ", ".join(c["runs"]))
            parts.append("status: " + c["status"])
            row[tool] = "; ".join(parts)
            short[tool + "_short"] = c["short"]
            note = c["note"]
            if tool == "lama" and any("exp05" in r for r in c["runs"]):
                note = (note + "; " if note else "") + "exp05 rows are the lama_adv arm (LAMA reader plus LGBAdvancedPipeline, not the full TabularAutoML)"
            if tool == "lama" and fam == "categorical_encoding":
                note += "; exp05_report.md flags 8% of lama_adv target-encoded columns on binary tasks against 1.5% of raw columns"
            ev.append({"family": fam, "tool": tool, "does_it": c["does"], "status": c["status"],
                       "ref_package": "|".join(pk.split("==")[0] for pk, _, _ in res),
                       "ref_version": "|".join(pk.split("==")[1] for pk, _, _ in res),
                       "ref_path": "|".join(p for _, p, _ in res), "ref_line": "|".join(str(ln) for _, _, ln in res),
                       "run_artifact": "|".join(c["runs"]), "note": note})
        rows.append({**row, **short})
    cols = ["family"] + TOOLS + [t + "_short" for t in TOOLS]
    out = RESULTS / "tidy"
    with open(out / "transform_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out / "transform_map_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "tool", "does_it", "status", "ref_package", "ref_version", "ref_path", "ref_line", "run_artifact", "note"])
        w.writeheader()
        w.writerows(ev)
    print(f"transform_map.csv, transform_map_evidence.csv: {len(rows)} families by {len(TOOLS)} tools")


if __name__ == "__main__":
    render()
