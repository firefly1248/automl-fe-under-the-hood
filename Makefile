ENVS := core autogluon lama openfe autofeat mljar
RUN = uv run --frozen --project envs/$(1)

.PHONY: env env-check data test exp01 exp03

env:
	for e in $(ENVS); do uv sync --frozen --project envs/$$e || exit 1; done

env-check:
	$(call RUN,core) python -m afe.envcheck report

data:
	$(call RUN,core) python -m afe.data.registry

test:
	$(call RUN,core) pytest -q tests

# exp01 step 4 and exp03 step 2 need `make data` first
exp01:
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/inventory.py
	$(call RUN,autogluon) python experiments/exp01_autogluon_probe/run.py

exp03:
	$(call RUN,lama) python experiments/exp03_lama_probe/run.py
