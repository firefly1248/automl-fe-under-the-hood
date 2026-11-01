ENVS := core autogluon lama openfe autofeat mljar
RUN = uv run --frozen --project envs/$(1)

.PHONY: env env-check data splits test exp01 exp03 exp04-smoke exp04-full

env:
	for e in $(ENVS); do uv sync --frozen --project envs/$$e || exit 1; done

env-check:
	$(call RUN,core) python -m afe.envcheck report

data:
	$(call RUN,core) python -m afe.data.registry

splits:
	$(call RUN,core) python -m afe.data.splits

test:
	$(call RUN,core) pytest -q tests
	for e in $(filter-out core,$(ENVS)); do AFE_ENV=$$e uv run --frozen --project envs/$$e pytest -q tests/test_generators_contract.py tests/test_openfe_equivalence.py || exit 1; done

# exp01 step 4 and exp03 step 2 need `make data` first
exp01:
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/inventory.py
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/run.py

exp03:
	$(call RUN,lama) python experiments/exp03_lama_probe/run.py

EXP04 = experiments/exp04_open_benchmark/config.yaml

exp04-smoke:
	$(call RUN,core) python -m afe.run.cli $(EXP04) smoke $(ARGS)

exp04-full:
	$(call RUN,core) python -m afe.run.cli $(EXP04) full $(ARGS)
