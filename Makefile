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
PY_VERSION := 3.13
PYTHONPATH := $(ROOT_DIR)

.PHONY: deps process-fullerenes run-demo check-cuda env env-cpu \
	clean distclean docker jupyter marimo voila jupyter-clean \
	nb-clean streamlit gaston docker-build docker-run help

help: ## Show available make targets
	@echo "EEGL Make targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.\/%-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(ENV): ## Create local Python environment
	bash scripts/python-env.sh create

check-cuda: ## Check CUDA/GPU environment info
	bash scripts/info.sh

env: $(ENV) ## Update Python environment
	bash scripts/python-env.sh update

env-cpu: ## Create/update CPU-only Python environment
	bash scripts/python-env.sh create cpu
	bash scripts/python-env.sh update cpu

clean: ## Remove temporary files and caches
	rm -rf *~ __pycache__

distclean: clean ## Remove temporary files and virtual environment
	rm -rf $(ENV)

docker-%: ## Run docker helper command (e.g. make docker-build)
	$(eval command = $(@:docker-%=%))
	bash scripts/docker.sh $(command)

jupyter: ## Launch Jupyter Lab
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

marimo: ## Launch Marimo lab
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

voila: ## Launch Voila server
	PYTHONPATH=$(ROOT_DIR) $(PY) -m voila --no-browser \
		--port=8866 \
		--no-browser --autoreload=true

jupyter-clean: nb-clean ## Alias for notebook cleanup

nb-clean notebook-clean: ## Clear notebook outputs
	@bash scripts/notebooks.sh clear-output

streamlit: ## Launch Streamlit app
	$(PY) -m streamlit run apps/web/streamlit/app.py

gaston: ## Install third-party Gaston dependency
	bash scripts/install-3rd-party.sh

deps: ## Sync project dependencies
	uv sync --no-dev

process-fullerenes: ## Run fullerene processing pipeline
	uv run -m apps.process_fullerenes

run-demo: ## Run demo workflow configuration
	uv run -m workflows.run -c run_configs/dev.yml \
		--run-defaults=run_configs/run_defaults.yml