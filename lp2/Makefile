PY=python3

setup:
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -e .
	bash scripts/setup_spacy.sh


fmt:
	ruff check --fix .
	ruff format .

lint:
	ruff check .

type:
	mypy lpa2

test:
	pytest

run:
	$(PY) -m lpa2.runners.generate --cfg lpa2/configs/default.yaml

eval:
	$(PY) -m lpa2.runners.evaluate --run_dir artifacts/runs/$(RUN)

ablate:
	$(PY) -m lpa2.runners.ablate --cfg lpa2/configs/default.yaml \
		--presets lpa_early_style_late,lpa_mid_only,lpa_late_only,lpa_full \
		--windows 200-700,300-650,400-600 --parsers spacy,pos,naive --n_prompts 20

figures:
	$(PY) -m lpa2.runners.plots --profiles artifacts/runs/*/profile.json \
		--out artifacts/figures/runtime.png || true
