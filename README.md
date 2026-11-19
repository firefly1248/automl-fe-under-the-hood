# automl-fe-under-the-hood

Experiments and numbers behind the Medium article "What automated feature engineering really does under the
hood" (sections and claim ids in [article/README.md](article/README.md)). Open data (OpenML) and open code only.

The question: what the feature generators of AutoGluon, LightAutoML, OpenFE, autofeat, MLJAR and skrub
actually do, and whether their features help a downstream LightGBM or CatBoost. The benchmark has 50 primary
OpenML datasets (20 binary, 10 multiclass, 20 regression), 5-fold CV repeated twice, one generator fit on the
training rows of each split. Case studies and planted-signal frames sit outside the primary set.

Every number and figure in the article is an entry of [results/claims.yaml](results/claims.yaml). `make claims`
recomputes each from the committed tidy tables and fails if one differs from its approved value. The plan is in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), the dated log in [PROGRESS.md](PROGRESS.md).

## Experiments

Each folder in `experiments/` has a README with the question, method, findings and caveats.

| folder | what it does |
|---|---|
| exp01_autogluon_probe | what AutoGluon's feature generators and `_r13` preprocessing produce |
| exp02_autogluon_ablation | AutoGluon ablation: features, hyperparameter groups, run-to-run noise |
| exp03_lama_probe | what LightAutoML's feature pipelines do |
| exp04_open_benchmark | the main benchmark: nine generator arms against the base, two downstream models |
| exp05_leakage_checks | permuted labels, transductive fit, planted signals |
| exp06_automl_end_to_end | AutoGluon, LightAutoML and H2O end to end |
| exp07_ctr_ablation | CatBoost target statistics ablation on the exp04 features |
| exp08_skrub_probe | what the skrub `TableVectorizer` defaults do to each column |
| exp09_numcast | diagnostics for categorical handling: number-like categoricals, integer codes |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and, on macOS, `brew install libomp`. Python 3.11; each tool has its own
pinned env in `envs/<tool>/` (`pyproject.toml` and `uv.lock`), and `src/afe` is installed editable in all of them.
LightAutoML 0.4.2 pulls in `torch`; `envs/lama` takes it from the CPU index on Linux (plain `pip install
lightautoml` was OOM-killed on an 8 GB machine) and from PyPI on macOS.

```
make env         # all envs
make env-check   # writes results/ENVIRONMENT.md
make test
```

## Reproduce

The run records are committed, so the analysis needs no reruns:

```
make unpack      # per-unit records from results/raw/<exp>/{gen,fit}.jsonl.gz
make aggregate figures article
make claims      # recompute every claim and compare with the approved values
```

Rerunning needs `make data` first (downloads the datasets to ~/.cache/afe, writes results/datasets.md), then
`make exp01 ... exp09` (targets in the Makefile, runtimes and run order in the experiment READMEs). The H2O arms
of exp06 need `envs/h2o` and Java 17.

## Layout

```
src/afe/          generators (one wrapper per arm), downstream models, run store and worker, analysis
experiments/      one folder per experiment: config, run script, README
results/          raw records, tidy tables, figures, claims.yaml, datasets.md, ENVIRONMENT.md
article/          article sections mapped to experiments and claim ids
tests/
```

MIT licence.
