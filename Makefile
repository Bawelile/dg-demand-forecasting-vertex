# Shortcuts. Run `make all` to do the whole training pipeline on your laptop.
PY ?= python

install:          ## install the project and test tools
	$(PY) -m pip install -e ".[dev]"

data:             ## Lesson 2: generate synthetic sales
	$(PY) -m dgforecast.data.generate
validate:         ## Lesson 3: check the data
	$(PY) -m dgforecast.data.validate
features:         ## Lesson 4: build leak-free features
	$(PY) -m dgforecast.features.build
train:            ## Lesson 5: train XGBoost
	$(PY) -m dgforecast.train.train
evaluate:         ## Lesson 6: score vs baseline and champion, gate, register
	$(PY) -m dgforecast.evaluate.evaluate
forecast:         ## Lesson 7: 14-day batch forecast with reasons
	$(PY) -m dgforecast.predict.batch
explain:          ## Lesson 8: what drives the forecast
	$(PY) -m dgforecast.explain.explain

all:              ## Lessons 2-8 in one go
	$(PY) -m dgforecast.pipeline.local_run
monitor:          ## Lessons 9-10: four weeks later: monitor, alert, retrain
	$(PY) -m dgforecast.pipeline.monitor_and_retrain

dashboard:        ## build dashboard.html from the latest results
	$(PY) -m dgforecast.report.dashboard

test:             ## run the tests
	$(PY) -m pytest -q
compile:          ## Lesson 10: compile the Vertex AI pipelines to YAML
	$(PY) -m dgforecast.pipeline.vertex_pipeline

clean:            ## delete all generated data and models
	rm -rf artifacts pipelines

.PHONY: install data validate features train evaluate forecast explain all monitor dashboard test compile clean
