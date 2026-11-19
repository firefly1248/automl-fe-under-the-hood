ENVS := core autogluon lama openfe autofeat mljar skrub
RUN = uv run --frozen --project envs/$(1)

.PHONY: exp08 exp02-noise exp06-noise exp06 exp07 aggregate figures article claims commit-results env env-check data planted splits test pack unpack exp01 exp02 exp03 exp03-step3 exp04-smoke exp04-pilot exp04-timing exp05 exp04-full-r0 exp04-full-r1

env:
	for e in $(ENVS); do uv sync --frozen --project envs/$$e || exit 1; done

env-check:
	$(call RUN,core) python -m afe.envcheck report

data:
	$(call RUN,core) python -m afe.data.registry

# planted frames of exp05 only, without the OpenML calls of `make data` (which also writes them)
planted:
	$(call RUN,core) python -m afe.data.registry planted

splits:
	$(call RUN,core) python -m afe.data.splits

test:
	$(call RUN,core) pytest -q tests
	for e in $(filter-out core,$(ENVS)); do AFE_ENV=$$e uv run --frozen --project envs/$$e pytest -q tests/test_generators_contract.py tests/test_openfe_equivalence.py || exit 1; done

# run records: per-unit JSON <-> committed results/raw/<exp>/{gen,fit}.jsonl.gz
pack:
	$(call RUN,core) python -m afe.run.pack pack

unpack:
	$(call RUN,core) python -m afe.run.pack unpack

# exp01 step 4 and exp03 step 2 need `make data` first
exp01:
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/inventory.py
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/run.py

# runs alone: one unit at a time with the whole machine (plan section 8)
exp02:
	$(call RUN,core) python experiments/exp02_autogluon_ablation/run.py $(ARGS)

# run-to-run noise repeats of D, E (exp02) and ag_best, lama, h2o (exp06) on pol and credit-g; one unit at a time
exp02-noise:
	$(call RUN,core) python experiments/exp02_autogluon_ablation/run.py --noise $(ARGS)

exp03:
	$(call RUN,lama) python experiments/exp03_lama_probe/run.py

exp03-step3:
	$(call RUN,lama) python experiments/exp03_lama_probe/fit_probe.py

EXP04 = experiments/exp04_open_benchmark/config.yaml

exp04-smoke:
	$(call RUN,core) python -m afe.run.cli $(EXP04) smoke $(ARGS)

exp04-pilot:
	$(call RUN,core) python -m afe.run.cli $(EXP04) pilot $(ARGS)

exp04-timing:
	$(call RUN,core) python -m afe.run.cli $(EXP04) timing --jobs 1 $(ARGS)

exp05:
	$(call RUN,core) python -m afe.run.cli experiments/exp05_leakage_checks/permuted.yaml run $(ARGS)
	$(call RUN,core) python -m afe.run.cli experiments/exp05_leakage_checks/transductive.yaml run $(ARGS)
	$(call RUN,core) python -m afe.run.cli experiments/exp05_leakage_checks/planted.yaml run $(ARGS)

exp04-full-r0:
	$(call RUN,core) python -m afe.run.cli $(EXP04) full_r0 $(ARGS)

exp04-full-r1:
	$(call RUN,core) python -m afe.run.cli $(EXP04) full_r1 $(ARGS)

# phase 6 (plan section 9)
aggregate:
	$(call RUN,core) python -m afe.analysis.aggregate
	$(call RUN,core) python -m afe.analysis.stats
	$(call RUN,core) python experiments/exp04_open_benchmark/report.py
	$(call RUN,core) python -m afe.analysis.exp02
	$(call RUN,core) python -m afe.analysis.exp06
	$(call RUN,core) python -m afe.analysis.noise
	$(call RUN,core) python -m afe.analysis.exp07

figures:
	$(call RUN,core) python -m afe.analysis.figures

# PNGs for the Medium article (results/figures/article), after aggregate; the map tables are rebuilt from the installed
# tool envs first (make env)
article:
	$(call RUN,core) python -m afe.analysis.transform_map_build
	$(call RUN,core) python -m afe.analysis.transform_map
	$(call RUN,core) python -m afe.analysis.article

claims:
	$(call RUN,core) python -m afe.analysis.claims

# stage everything reviewable; shows the list, commits nothing (plan rule 9)
commit-results: aggregate figures article claims pack
	git add -A results experiments article README.md PROGRESS.md
	git status --short

# optional, runs alone, after exp02 (A4 in PROGRESS.md 2026-11-10); needs envs/h2o and Java 17
exp06:
	$(call RUN,core) python experiments/exp06_automl_end_to_end/run.py $(ARGS)

exp06-noise:
	$(call RUN,core) python experiments/exp06_automl_end_to_end/run.py --noise $(ARGS)

# CatBoost max_ctr_complexity=1 ablation: downstream fits on the cached exp04 generator outputs (680 units)
exp07:
	$(call RUN,core) python experiments/exp07_ctr_ablation/run.py $(ARGS)

# skrub TableVectorizer probe (A4, 2026-11-17): what its defaults do to each column
exp08:
	$(call RUN,skrub) python experiments/exp08_skrub_probe/run.py
