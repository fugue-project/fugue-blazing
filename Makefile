.PHONY: help clean dev docs package test

help:
	@echo "The following make targets are available:"
	@echo "	 devenv		create venv and install all deps for dev env (assumes python3 cmd exists)"
	@echo "	 dev 		install all deps for dev env (assumes venv is present)"
	@echo "	 package	package for pypi"
	@echo "	 test		run all tests with coverage (assumes venv is present)"

devenv:
	pip3 install -r requirements.txt
	pre-commit install

dev:
	pip3 install -r requirements.txt

lint:
	pre-commit run --all-files

package:
	rm -rf dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test:
	python3 -bb -m pytest tests/

jupyter:
	pip install .
	rm -rf /root/.jupyter/jupyter_config.py
	rm -rf /root/.ipython/profile_default/startup/
	mkdir -p /root/.ipython/profile_default/startup/
	cp fugue_notebook/jupyter_config.py /root/.ipython/profile_default/startup/
	jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
