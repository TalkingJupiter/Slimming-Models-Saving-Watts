SHELL := /usr/bin/env bash

TEACHER ?= meta-llama/Meta-Llama-3.1-70B-Instruct
STUDENT ?= meta-llama/Meta-Llama-3.1-8B-Instruct
SAFE_TEACHER_NAME ?= $(subst /,_,$(TEACHER))
SAFE_STUDENT_NAME ?= $(subst /,_,$(STUDENT))
TEACHER_DATA ?= $(SAFE_TEACHER_NAME)
PIPELINE_TARGET ?= all

.PHONY: help submit submit-local clean clean-logs clean-results clean-cache clean-teacher-cache clean-teacher-caches clean-shards clean-all-generated show-paths

help:
	@echo "Targets:"
	@echo "  make submit              Submit submit_h100.sh with sbatch"
	@echo "  make submit-local        Run submit_h100.sh directly"
	@echo "  make clean               Remove logs, results, and data/\$$(TEACHER_DATA) cache outputs"
	@echo "  make clean-logs          Remove logs/"
	@echo "  make clean-results       Remove results/"
	@echo "  make clean-cache         Alias for clean-teacher-cache"
	@echo "  make clean-teacher-cache Remove data/\$$(TEACHER_DATA)"
	@echo "  make clean-teacher-caches Alias for clean-teacher-cache"
	@echo "  make clean-shards        Remove data/shards.jsonl"
	@echo "  make clean-all-generated Remove logs, results, data/\$$(TEACHER_DATA), and data/shards.jsonl"
	@echo ""
	@echo "Variables:"
	@echo "  TEACHER=$(TEACHER)"
	@echo "  STUDENT=$(STUDENT)"
	@echo "  TEACHER_DATA=$(TEACHER_DATA)"
	@echo "  PIPELINE_TARGET=$(PIPELINE_TARGET)"
	@echo ""
	@echo "Examples:"
	@echo "  make submit"
	@echo "  make submit TEACHER=\"meta-llama/Meta-Llama-3.1-70B-Instruct\" STUDENT=\"meta-llama/Meta-Llama-3.1-8B-Instruct\""
	@echo "  make submit-local TEACHER_DATA=\"meta-llama_Meta-Llama-3.1-70B-Instruct\""
	@echo "  make show-paths TEACHER=\"meta-llama/Meta-Llama-3.1-70B-Instruct\""
	@echo "  make clean TEACHER=\"meta-llama/Meta-Llama-3.1-70B-Instruct\""
	@echo "  make clean-cache TEACHER_DATA=\"meta-llama_Meta-Llama-3.1-70B-Instruct\""
	@echo "  make clean-all-generated TEACHER_DATA=\"meta-llama_Meta-Llama-3.1-70B-Instruct\""

show-paths:
	@echo "logs:          logs"
	@echo "results:       results"
	@echo "teacher cache: data/$(TEACHER_DATA)"
	@echo "shards:        data/shards.jsonl"

submit:
	@mkdir -p logs/h100
	TEACHER="$(TEACHER)" STUDENT="$(STUDENT)" TEACHER_DATA="$(TEACHER_DATA)" sbatch submit_h100.sh "$(PIPELINE_TARGET)"

submit-local:
	@mkdir -p logs/h100
	TEACHER="$(TEACHER)" STUDENT="$(STUDENT)" TEACHER_DATA="$(TEACHER_DATA)" bash submit_h100.sh "$(PIPELINE_TARGET)"

clean: clean-logs clean-results clean-teacher-cache

clean-logs:
	$(RM) -r logs

clean-results:
	$(RM) -r results

clean-teacher-cache:
	@if [[ -z "$(TEACHER_DATA)" || "$(TEACHER_DATA)" == "." || "$(TEACHER_DATA)" == "/" ]]; then \
		echo "[ERROR] Refusing to remove unsafe TEACHER_DATA='$(TEACHER_DATA)'." >&2; \
		exit 2; \
	fi
	$(RM) -r "data/$(TEACHER_DATA)"

clean-cache clean-teacher-caches: clean-teacher-cache

clean-shards:
	$(RM) -f data/shards.jsonl

clean-all-generated: clean clean-shards
