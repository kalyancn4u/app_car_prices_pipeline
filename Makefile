# Convenience targets for the full lifecycle. On Windows, run these commands
# directly if you don't have `make` (see the comment on each target).

PY ?= python

.PHONY: setup train test notebooks predict clean

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
