# Project Completion Plan

## Goal

Turn `meshmerizer` from a well-refactored codebase into a fully rounded,
releaseable project with:

- reliable CI,
- a documented release process,
- safe PyPI publishing,
- complete project metadata,
- contributor-facing guidance,
- a real documentation site, and
- a clear packaging strategy for the native extension.

This plan is intentionally practical and ordered so each stage reduces risk for
the next one.

---

## Guiding principles

- Prefer automation over manual release steps.
- Validate installs from built artifacts, not only editable installs.
- Treat the native extension as a first-class packaging concern.
- Keep TestPyPI and PyPI publishing paths explicit and separate.
- Keep repository metadata and documentation aligned with the actual package
  structure.

---

## Workstreams

### 1. Finalize package metadata and repository essentials

**Goal:** make the package metadata complete and trustworthy before adding more
automation.

**Current gaps:**

- `pyproject.toml` is missing `project.urls`
- there is no checked-in `LICENSE` file
- classifiers and metadata are still fairly minimal

**Changes:**

- Add a top-level `LICENSE` file
- Add `project.urls` to `pyproject.toml`, including at least:
  - homepage
  - repository
  - issues
  - documentation
  - changelog
- Review and tighten classifiers
- Decide whether the long description / package description should be updated to
  reflect the current architecture and native meshing pipeline
- Change the console script entrypoint from:
  - `meshmerizer.commands.main:main`
  to:
  - `meshmerizer.cli.main:main`

**Deliverables:**

- `LICENSE`
- updated `pyproject.toml`

**Validation:**

- `python -m build`
- install the built wheel in a clean environment
- verify `meshmerizer --help` works from the installed artifact

---

### 2. Add core project governance files

**Goal:** make the repository usable by future contributors and by your future
self.

**Current gaps:**

- no `CHANGELOG.md`
- no `CONTRIBUTING.md`
- no dedicated development workflow document

**Changes:**

- Add `CHANGELOG.md`
- Add `CONTRIBUTING.md`
- Optionally add `DEVELOPMENT.md` if the contributor guide becomes too large
- Document:
  - local install flow
  - lint/test commands
  - how the native extension is built
  - how to run targeted tests
  - how releases are cut

**Deliverables:**

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- optional `DEVELOPMENT.md`

**Validation:**

- docs are internally consistent with `README.md` and `pyproject.toml`
- all listed commands run successfully

---

### 3. Establish a release and versioning policy

**Goal:** define how versions are created and how releases move from git to
package indexes.

**Current gaps:**

- versioning is dynamic via `setuptools_scm`, but release workflow is not yet
  defined
- no tag/release policy is written down
- no changelog workflow is defined

**Changes:**

- Choose and document a release model, e.g.:
  - tag-driven releases with `setuptools_scm`
  - semantic versioning or semver-like policy
- Define how changelog entries are maintained
- Define the exact release checklist:
  - merge to main
  - confirm CI green
  - create release tag
  - publish to TestPyPI (optional gate)
  - publish to PyPI
  - create GitHub release notes

**Deliverables:**

- release section in `CONTRIBUTING.md` or `DEVELOPMENT.md`
- initial `CHANGELOG.md` structure

**Validation:**

- cut a dry-run pre-release tag locally or in a test branch
- confirm `setuptools_scm` resolves the expected version from tags

---

### 4. Add continuous integration

**Goal:** ensure every change is automatically checked for build, lint, and test
health.

**Current gaps:**

- CI exists, but build-artifact validation is still incomplete

**Changes:**

- Add GitHub Actions workflows for:
  - linting (`ruff check`)
  - tests (`pytest`)
  - build verification (`python -m build`)
  - artifact sanity (`twine check dist/*`)
- Start with a pragmatic matrix, for example:
  - Python 3.8+
  - Linux and macOS
- Decide whether native-extension-heavy jobs should be split from pure lint jobs

**Deliverables:**

- `.github/workflows/ci.yml`
- optional separate workflow files for lint/build/release

**Validation:**

- CI passes on a clean PR
- wheel/sdist artifacts build successfully in CI
- package metadata passes `twine check`

---

### 5. Define and implement the packaging strategy for the native extension

**Goal:** make installation expectations explicit for users installing from PyPI.

**Current gaps:**

- native extension packaging strategy is not yet formalized
- unclear whether PyPI distribution will be sdist-only or include wheels

**Decision required:**

Choose one of:

#### Option A: sdist-first

Publish source distributions only and require local compilation.

**Pros:**

- simplest release machinery
- less CI complexity

**Cons:**

- worse user install experience
- greater toolchain burden on users

#### Option B: sdist + wheels

Publish wheels for supported platforms using `cibuildwheel`.

**Pros:**

- much better install experience
- realistic for a native extension package intended for wider use

**Cons:**

- more CI and release complexity
- platform coverage decisions become important

**Recommended direction:**

For this project, Option B is likely the better end state.

**Changes:**

- Audit build inputs to ensure all native sources are packaged correctly
- Add wheel build configuration
- If using wheels, add `cibuildwheel`
- Decide supported targets, e.g.:
  - macOS x86_64 / arm64
  - Linux x86_64
- Decide whether Windows is supported or explicitly unsupported for now

**Deliverables:**

- packaging notes in docs
- optional `cibuildwheel` configuration in CI

**Validation:**

- install built wheel in a clean env
- run a smoke test:
  - import package
  - run CLI help
  - run a tiny reconstruction test

---

### 6. Add PyPI and TestPyPI publishing automation

**Goal:** make releases safe, repeatable, and low-friction.

**Current gaps:**

- PyPI publishing exists, but a TestPyPI path is still not defined

**Recommended approach:**

Use **GitHub Actions + PyPI trusted publishing** rather than a local deploy
script as the primary publishing path.

You can still provide small local validation helpers if needed, but the
canonical release path should be CI-driven.

**Changes:**

- Configure a TestPyPI publish workflow
- Configure a PyPI publish workflow
- Use trusted publishing if possible
- Restrict PyPI publishing to:
  - tags only
  - or GitHub release events only
- Add a local helper script only if it genuinely adds value beyond CI.

**Deliverables:**

- `.github/workflows/publish-testpypi.yml`
- `.github/workflows/publish-pypi.yml`
- optional `scripts/` helpers

**Validation:**

- successful upload to TestPyPI
- successful install from TestPyPI in a clean environment
- successful smoke test from TestPyPI artifact

---

### 7. Add built-artifact install and smoke-test validation

**Goal:** ensure the packaged project actually works when installed the way
users will install it.

**Current gaps:**

- current workflow appears focused on editable installs
- no explicit smoke-test of built artifacts is documented

**Changes:**

- Add a CI job that:
  - builds wheel/sdist
  - installs from the built artifact into a fresh virtual environment
  - runs a minimal smoke test
- Smoke test should include:
  - `import meshmerizer`
  - `meshmerizer --help`
  - one tiny Python API invocation if practical

**Deliverables:**

- CI artifact-install job
- documented smoke-test commands

**Validation:**

- green install-validation job in CI

---

### 8. Build a real documentation site

**Goal:** move from README-only docs to a proper documentation structure.

**Current gaps:**

- the docs site exists, but it still needs broader contributor-facing and
  release/process coverage

**Recommended approach:**

Use **MkDocs**, ideally with **Material for MkDocs**.

This is a good fit because the project needs:

- narrative docs,
- CLI reference pages,
- API usage docs,
- architecture docs,
- developer/native-core notes.

**Changes:**

- Add `mkdocs.yml`
- Add docs structure, likely including:
  - `docs/index.md`
  - `docs/installation.md`
  - `docs/cli.md`
  - `docs/python-api.md`
  - `docs/architecture.md`
  - `docs/native-core.md`
  - `docs/contributing.md`
  - `docs/releases.md`
- Decide whether to use an API doc plugin or keep API docs hand-written
- Add local docs serve/build commands
- Optionally add GitHub Pages deployment

**Deliverables:**

- `mkdocs.yml`
- docs site content under `docs/`

**Validation:**

- `mkdocs serve`
- `mkdocs build`
- links resolve and navigation is coherent

---

### 9. Add issue/PR hygiene and repository templates

**Goal:** make repository collaboration cleaner once the project is public or
shared more broadly.

**Current gaps:**

- no issue templates
- no PR template
- no CODEOWNERS

**Changes:**

- Add issue templates for:
  - bug reports
  - feature requests
- Add PR template
- Optionally add `CODEOWNERS`

**Deliverables:**

- `.github/ISSUE_TEMPLATE/...`
- `.github/pull_request_template.md`
- optional `.github/CODEOWNERS`

**Validation:**

- templates appear correctly in GitHub UI

---

### 10. Add optional quality-of-life tooling

**Goal:** make development and maintenance smoother without blocking releases.

**Potential additions:**

- `pre-commit`
- `CITATION.cff` for research use
- benchmark/performance notes
- example notebooks or example commands
- release helper scripts

**Recommended priority:** low after CI/docs/release automation are in place.

---

## Recommended execution order

1. package metadata and essentials
2. contributor/governance files
3. release/versioning policy
4. CI
5. native-extension packaging strategy
6. TestPyPI/PyPI publishing automation
7. built-artifact smoke testing
8. MkDocs documentation site
9. GitHub templates and hygiene files
10. optional quality-of-life tooling

---

## Immediate next actions

If starting now, the highest-value next steps are:

1. add `LICENSE`
2. update `pyproject.toml` metadata and entrypoint
3. add `CHANGELOG.md` and `CONTRIBUTING.md`
4. add basic CI (`ruff`, `pytest`, `build`, `twine check`)
5. choose wheel strategy for the native extension
6. then build the MkDocs site and publishing workflows

---

## Success criteria

The repository can be considered a complete project when:

- it has a proper license file and complete package metadata
- contributors can follow a documented local development workflow
- releases follow a documented versioning and tagging policy
- CI automatically validates lint, tests, and package builds
- built artifacts are install-tested in a clean environment
- TestPyPI and PyPI publishing are automated and safe
- the documentation site covers installation, CLI, API, architecture, and
  development
- the native extension support story is explicit and validated
- repository templates and release hygiene are in place
