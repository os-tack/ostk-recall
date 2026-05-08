# ostk-recall — local-dev helpers.
#
# `make help` lists everything. The common loop is:
#   make install              # build + drop the binary into ~/.cargo/bin
#   make scan                 # ingest configured sources into the corpus
#   make verify               # sanity-check counts after a scan
#   make serve                # run the MCP endpoint
#
# Build artifacts live under ./target. The installed binary lives at
# $(CARGO_HOME)/bin/ostk-recall (default ~/.cargo/bin/ostk-recall).

SHELL          := /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c
.DEFAULT_GOAL  := help

BIN            := ostk-recall
CARGO          ?= cargo
CARGO_HOME     ?= $(HOME)/.cargo
INSTALLED_BIN  := $(CARGO_HOME)/bin/$(BIN)
RELEASE_BIN    := target/release/$(BIN)
CONFIG         ?= $(HOME)/.config/ostk-recall/config.toml
GIT_SHA        := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY      := $(shell git diff --quiet 2>/dev/null || echo "-dirty")
PKG_VERSION    := $(shell awk -F\" '/^version = / {print $$2; exit}' Cargo.toml 2>/dev/null || echo "?")

# Forward extra args to a target with `make scan -- --force`.
ARGS := $(filter-out $@,$(MAKECMDGOALS))

# ── meta ──────────────────────────────────────────────────────────────

.PHONY: help
help: ## list every documented target
	@awk 'BEGIN {FS = ":.*##"; printf "ostk-recall  v$(PKG_VERSION) ($(GIT_SHA)$(GIT_DIRTY))\n\nUsage:\n  make <target>\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: version
version: ## show installed binary version + path + mtime
	@printf "workspace:   v$(PKG_VERSION) ($(GIT_SHA)$(GIT_DIRTY))\n"
	@if [ -x "$(INSTALLED_BIN)" ]; then \
		printf "installed:   "; "$(INSTALLED_BIN)" --version; \
		printf "  path:      %s\n" "$(INSTALLED_BIN)"; \
		printf "  built:     "; stat -f '%Sm' "$(INSTALLED_BIN)" 2>/dev/null || stat -c '%y' "$(INSTALLED_BIN)"; \
	else \
		printf "installed:   <none — run 'make install'>\n"; \
	fi
	@if [ -x "$(RELEASE_BIN)" ]; then \
		printf "release:     "; "$(RELEASE_BIN)" --version; \
		printf "  path:      %s\n" "$(RELEASE_BIN)"; \
		printf "  built:     "; stat -f '%Sm' "$(RELEASE_BIN)" 2>/dev/null || stat -c '%y' "$(RELEASE_BIN)"; \
	fi

# ── build ─────────────────────────────────────────────────────────────

.PHONY: build
build: ## debug build (fast, with assertions)
	$(CARGO) build --bin $(BIN)

.PHONY: release
release: ## release build (slim, optimized) → target/release/ostk-recall
	$(CARGO) build --release --bin $(BIN)

.PHONY: install
install: ## build release + install to ~/.cargo/bin/ostk-recall
	$(CARGO) install --path crates/cli --locked --force
	@printf "\ninstalled: %s\n" "$(INSTALLED_BIN)"
	@$(INSTALLED_BIN) --version

.PHONY: uninstall
uninstall: ## remove ~/.cargo/bin/ostk-recall
	$(CARGO) uninstall $(BIN) || true

# ── operate ───────────────────────────────────────────────────────────
# These shell out to the *installed* binary so they exercise the same
# path any caller (MCP server, kernel, your shell) would hit. Use
# `make scan-local` to run from target/release/ instead.

.PHONY: scan
scan: $(INSTALLED_BIN) ## scan all configured sources (uses installed binary)
	$(INSTALLED_BIN) scan

.PHONY: scan-rebuild
scan-rebuild: $(INSTALLED_BIN) ## full rescan — purges + rebuilds the corpus
	$(INSTALLED_BIN) scan --force

.PHONY: scan-local
scan-local: release ## scan using the just-built target/release binary
	$(RELEASE_BIN) scan

.PHONY: verify
verify: $(INSTALLED_BIN) ## sanity-check corpus counts
	$(INSTALLED_BIN) verify

.PHONY: serve
serve: $(INSTALLED_BIN) ## run the MCP endpoint (foreground)
	$(INSTALLED_BIN) serve

.PHONY: inspect
inspect: $(INSTALLED_BIN) ## inspect a chunk by id: make inspect ID=<chunk_id>
	@if [ -z "$(ID)" ]; then echo "usage: make inspect ID=<chunk_id>"; exit 2; fi
	$(INSTALLED_BIN) inspect $(ID)

.PHONY: init
init: $(INSTALLED_BIN) ## create a fresh corpus from $(CONFIG)
	$(INSTALLED_BIN) init

# ── config ────────────────────────────────────────────────────────────

.PHONY: config-show
config-show: ## print the resolved config path + contents
	@printf "config: %s\n\n" "$(CONFIG)"
	@cat "$(CONFIG)"

.PHONY: config-edit
config-edit: ## open $(CONFIG) in $$EDITOR
	@$${EDITOR:-vi} "$(CONFIG)"

# ── quality gates ─────────────────────────────────────────────────────

.PHONY: test
test: ## run all workspace tests
	$(CARGO) test --workspace

.PHONY: lint
lint: ## clippy on the full workspace (warnings tolerated)
	$(CARGO) clippy --workspace --all-targets

.PHONY: lint-strict
lint-strict: ## clippy with -D warnings (CI mode)
	$(CARGO) clippy --workspace --all-targets -- -D warnings

.PHONY: fmt
fmt: ## auto-format
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## verify formatting without writing
	$(CARGO) fmt --all -- --check

.PHONY: check
check: fmt-check lint test ## fmt + clippy + tests (same gates as CI)

# ── housekeeping ──────────────────────────────────────────────────────

.PHONY: clean
clean: ## cargo clean
	$(CARGO) clean

# Auto-build-and-install when a target depends on $(INSTALLED_BIN) but
# the installed binary is older than any tracked source.
$(INSTALLED_BIN): $(shell find crates -name '*.rs' 2>/dev/null) Cargo.toml Cargo.lock
	@echo ">> rebuilding & installing $(BIN) (sources newer than $@)"
	@$(MAKE) install

# Swallow extra positional args so `make scan -- --force` doesn't error
# on the trailing words.
%:
	@:
