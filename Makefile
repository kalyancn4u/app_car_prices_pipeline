# Convenience targets for the full lifecycle. On Windows, run these commands
# directly if you don't have `make` (see the comment on each target).

PY ?= python

# --- Clean-rebuild pipeline (fresh conda env -> train -> verify -> pin) --------
# Mirrors the Streamlit app's discipline: the env that trains models/*.pkl also
# writes requirements.txt + .python-version, so any target installs the exact
# versions the pickle was saved with (no version-mismatch crash on load).
#
# Python version policy (why 3.11): the pins numpy 1.26.x / scikit-learn 1.6.x
# only have wheels for Python 3.9-3.12, so this project stays on 3.11. Do NOT
# pair these pins with Python 3.13/3.14 - no numpy 1.26 wheel exists there and
# the install fails. To move to 3.13+ you must upgrade the whole stack AND
# retrain (also check xgboost/lightgbm ship 3.13 wheels): unpin the core libs,
# `make rebuild PYVER=3.13`, then `make freeze`. See README -> "Reproducible
# rebuilds & the Python-version policy".
ENV   ?= car-pipeline
PYVER ?= 3.11
CONDA ?= conda
RUN   := $(CONDA) run -n $(ENV) --no-capture-output

.PHONY: setup train test notebooks predict clean env verify freeze rebuild push

setup:            ## Install the package + dependencies (editable)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

train:            ## Run the model bake-off + train/select/save the pipeline
	$(PY) -m car_pricing.train

test:             ## Run the test suite
	$(PY) -m pytest -q

notebooks:        ## (Re)generate the phase notebooks from tools/build_notebooks.py
	$(PY) tools/build_notebooks.py

predict:          ## Example prediction (edit the payload in this target)
	$(PY) -c "from car_pricing.predict import predict; print(predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"

clean:            ## Remove caches and generated processed data
	rm -rf **/__pycache__ .pytest_cache data/processed/_bench

env:              ## Create conda env (Python $(PYVER)) + install deps (editable)
	$(CONDA) create -y -n $(ENV) python=$(PYVER)
	$(RUN) python -m pip install --upgrade pip
	$(RUN) python -m pip install -r requirements.txt
	$(RUN) python -m pip install -e .

verify:           ## Confirm the saved pipeline unpickles AND predicts
	$(RUN) python -c "from car_pricing.predict import predict; print('OK:', predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"

freeze:           ## Pin requirements.txt + .python-version to this env
	$(RUN) python tools/pin_env.py requirements.txt

rebuild: env      ## Full clean run: env -> train -> verify -> pin
	$(RUN) python -m car_pricing.train
	$(RUN) python -c "from car_pricing.predict import predict; print('OK:', predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"
	$(RUN) python tools/pin_env.py requirements.txt
	@echo ""
	@echo "Rebuild complete. Review 'git diff', then 'make push'."

push:             ## Commit retrained model + pins and push
	git add models/ requirements.txt .python-version
	git commit -m "Rebuild pipeline and re-pin environment (Python $(PYVER))"
	git push origin HEAD
