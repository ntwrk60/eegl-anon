# EEGL - An Iterative GNN Enhancement Framework
# Copyright (C) 2025 The EEGL Authors

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see
# <https://www.gnu.org/licenses/>.

CURRENT_FILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(shell dirname $(CURRENT_FILE_PATH))

ENV := $(ROOT_DIR)/.venv
ENV_CPU := $(ROOT_DIR)/.venv
PY := $(ENV)/bin/python
PY_VERSION := 3.12
PYTHONPATH := $(ROOT_DIR)

.PHONY: deps process-fullerenes run-demo check-cuda env env-cpu \
	clean distclean docker jupyter marimo voila jupyter-clean \
	nb-clean streamlit gaston docker-build docker-run

$(ENV):
	bash scripts/python-env.sh create

check-cuda:
	bash scripts/info.sh

env: $(ENV)
	bash scripts/python-env.sh update

env-cpu:
	bash scripts/python-env.sh create cpu
	bash scripts/python-env.sh update cpu

clean:
	rm -rf *~ __pycache__

distclean: clean
	rm -rf $(ENV)

docker-%:
	$(eval command = $(@:docker-%=%))
	bash scripts/docker.sh $(command)

jupyter:
	PYTHONPATH=$(ROOT_DIR):$(ROOT_DIR)/apps/gnn_explainer \
		$(PY) -m jupyter lab \
		--no-browser \
		--ip=0.0.0.0 --port=8080 \
		--autoreload \
		--ServerApp.base_url=/egr \
		--IdentityProvider.token='' \
		--ServerApp.allow_origin='*' \
		--ServerApp.allow_remote_access=True \
		--ServerApp.disable_check_xsrf=True

marimo:
	PYTHONPATH=$(ROOT_DIR):$(ROOT_DIR)/apps/gnn_explainer \
		$(PY) -m marimo lab \
		--no-browser \
		--ip=0.0.0.0 --port=8080 \
		--autoreload \
		--ServerApp.base_url=/egr \
		--IdentityProvider.token='' \
		--ServerApp.allow_origin='*' \
		--ServerApp.allow_remote_access=True \
		--ServerApp.disable_check_xsrf=True

voila:
	PYTHONPATH=$(ROOT_DIR) $(PY) -m voila --no-browser \
		--port=8866 \
		--no-browser --autoreload=true

jupyter-clean: nb-clean

nb-clean notebook-clean:
	@bash scripts/notebooks.sh clear-output

streamlit:
	$(PY) -m streamlit run apps/web/streamlit/app.py

gaston:
	bash scripts/install-3rd-party.sh

deps:
	uv sync --no-dev

process-fullerenes:
	uv run -m apps.process_fullerenes

run-demo:
	uv run -m workflows.run -c run_configs/dev.yml \
		--run-defaults=run_configs/run_defaults.yml