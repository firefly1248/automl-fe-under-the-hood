ENVS := core autogluon lama openfe autofeat mljar
RUN = uv run --frozen --project envs/$(1)

.PHONY: env env-check data planted splits test pack unpack exp01 exp03 exp04-smoke exp04-pilot exp04-timing exp05 exp04-full-r0 exp04-full-r1

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

exp03:
	$(call RUN,lama) python experiments/exp03_lama_probe/run.py

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
