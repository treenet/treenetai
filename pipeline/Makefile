
# Makefile — docs utilities for TreeNet pre-processing

.PHONY: help docs docs-md clean-docs

# Default module to document
MODULE ?= build_normalized_dataset_treenet.py
OUT_MD ?= $(basename $(MODULE))_API.md

help:
	@echo "Targets:"
	@echo "  make docs        - Generate Markdown API docs for $(MODULE) -> $(OUT_MD)"
	@echo "  make docs-md     - Same as 'docs' (explicit)"
	@echo "  make clean-docs  - Remove generated *_API.md files"
	@echo "Variables:"
	@echo "  MODULE=<path>    - Module to document (default: build_normalized_dataset_treenet.py)"
	@echo "  OUT_MD=<path>    - Output Markdown path (default: <module>_API.md)"

# Primary docs target
 docs: make_docs.py $(MODULE)
	python make_docs.py $(MODULE) --out $(OUT_MD)

# Alias
 docs-md: docs

clean-docs:
	rm -f *_API.md
