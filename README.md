# automl-fe-under-the-hood

Work in progress. Numbers are not final; approved values will be in results/claims.yaml.

Experiments and numbers behind the LinkedIn series "What automated feature engineering really does
under the hood". Open data (OpenML) and open code only. The plan is in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), progress in [PROGRESS.md](PROGRESS.md).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and, on macOS, `brew install libomp`.

```
make env         # one pinned env per tool in envs/, Python 3.11
make env-check   # writes results/ENVIRONMENT.md
make data        # downloads datasets to ~/.cache/afe, writes results/datasets.md
make exp01 exp03
make unpack      # per-unit run records from the committed results/raw/<exp>/{gen,fit}.jsonl.gz
```

## Environments

Each tool runs in its own env (`envs/<tool>/pyproject.toml` + `uv.lock`); `src/afe` is installed
editable in every env. LightAutoML 0.4.2 pulls in `torch`; `envs/lama` takes torch from the CPU index
on Linux (plain `pip install lightautoml` was OOM-killed on an 8 GB machine), PyPI torch on macOS.
