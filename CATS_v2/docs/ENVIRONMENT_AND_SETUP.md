# CATS v2 Environment and Setup Contract

**Status:** Current setup guide for local development, remote evaluation,
local-committee evaluation, human review, and workbook export.

## 1. Environment layers

The repository has four related but distinct environment needs:

| Layer | Purpose | Main dependency source |
| --- | --- | --- |
| Core evaluator | Read JSONL, normalize outputs, run metrics, call configured judges, and write reports. | Root `requirements.txt` |
| Local committee serving | Host Qwen, Mistral, and DeepSeek-compatible OpenAI endpoints on Sharanga. | Sharanga environment and Slurm serving scripts |
| Human-evaluation CLI | Run reviewer sessions and export human judgments. | `exports/cats_human_eval_cli/pyproject.toml` |
| Workbook export | Read/update the hierarchical XLSX and verify cells. | `openpyxl` plus the updater's configured dependency path |

These layers should not be collapsed into one undocumented environment. A
reviewer can inspect the repository, metrics, and existing outputs without
running GPU serving.

## 2. Core Python environment

The root dependency manifest is
[`../requirements.txt`](../requirements.txt). It covers HTTP/API clients,
NumPy/Pandas/SciPy, NLTK and text utilities, async/concurrency helpers,
configuration, logging, retry, and optional model/NLI support.

The code requires Python 3.8 or newer according to `test_installation.py`.
The standalone human package requires Python 3.11 or newer according to its
`pyproject.toml`; use a separate environment if the host Python is older.

Recommended development setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository ignores `venv/`, `__pycache__/`, and `.pyc` files. Do not commit
secrets or local virtual-environment directories.

## 3. NLTK data

The metric code has a safe sentence-tokenization fallback, but the installation
smoke test expects the NLTK tokenizer data to be available. If the test reports
a tokenizer failure, install the required data in the active environment:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

The NLTK data location is environment-specific and is not part of the result
artifact. Record the Python/environment identity when reporting a fresh run.

## 4. Secrets and provider selection

`.env.example` documents legacy and alternate provider keys. Copy it locally:

```bash
cp .env.example .env
```

Never commit `.env`. The relevant key depends on the selected committee:

| Run mode | Required secret |
| --- | --- |
| `--committee default` with OpenRouter judges | `OPENROUTER_API_KEY` |
| Direct DeepSeek judge or mixed committee | `DEEPSEEK_API_KEY` |
| `--committee local` | No API key for local endpoints; endpoint reachability is required. |
| `--committee cli` | Codex CLI authentication/configuration outside this repository. |
| Standalone legacy NLI path | Provider key specified by the active NLI config. |

Run `test_api_keys.py` only when remote providers are intended. It performs live
network calls and is not required for local committee or offline read-only cache
aggregation. Do not print full keys in logs or reports.

## 5. Installation and smoke checks

Run the repository installation check after dependency setup:

```bash
python test_installation.py
```

It checks Python version, core imports, `rag_eval` imports, expected directories,
`.env` visibility, NLTK tokenization, and canonical benchmark/split assets. A
warning about a missing remote key is expected when only the local committee is
being used, but a missing core import or canonical data asset is a blocker.

The code-only regression check is:

```bash
python3 -m unittest discover -s tests -q
python3 -m py_compile rag_eval/*.py run_evaluation.py scripts/*.py
```

These checks do not call model endpoints and do not certify that a live local
server is healthy.

## 6. Local committee environment

Local judges expose an OpenAI-compatible endpoint with:

```text
GET /v1/models
POST /v1/chat/completions
```

The benchmark YAML supplies model id, base URL, priority, maximum output tokens,
timeout, and optional model-specific request fields. Servers are launched on
Sharanga through the scripts under
[`../slurm/sharanga/local_committee/`](../slurm/sharanga/local_committee/).

The serving environment, model weights, CUDA modules, GPU placement, chat
templates, and model-local staging are cluster-specific. They are intentionally
not installed by the root `requirements.txt`. Read the local committee guide
and serving README before launching a model.

Endpoint readiness requires a successful JSON completion probe, not merely a
running Slurm job or a successful `/models` response.

## 7. Human-evaluation environment

The standalone package declares only its reviewer-side dependencies:

```bash
cd exports/cats_human_eval_cli
python3.11 -m venv .venv-human
source .venv-human/bin/activate
python -m pip install -e .
python -m cats_human_eval --help
```

Reviewer bundles are intended to run independently of the root evaluator. The
study data, assignments, SQLite state, event log, and exports travel with the
bundle. Do not make reviewers install the full GPU/committee environment.

The human package's current Python contract is `>=3.11`, with Rich, Typer, and
PyYAML. The authoritative workflow and receipt rules are in
[`HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md`](HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md).

## 8. Workbook environment

The hierarchical workbook updater imports `openpyxl`. In the current local
workspace, `scripts/update_master_results_workbook.py` adds
`/private/tmp/cats_excel_deps` to `sys.path` before importing it. That path is a
machine-local implementation detail, not a portable repository dependency.

Before reproducing workbook export on another machine, ensure a compatible
`openpyxl` installation is available and either provide the configured helper
path or adapt the environment without changing the workbook logic. Then run:

```bash
python scripts/update_master_results_workbook.py --help
```

Always write to a new output workbook and audit JSON. Do not overwrite the
source workbook until row matching, header preservation, cell verification, and
formatting checks have passed.

## 9. Offline versus online operations

| Operation | Needs network/model server? |
| --- | --- |
| Read existing detailed JSON and recompute CATS | No. |
| Run unit tests and compile checks | No. |
| Human receipt consolidation | No. |
| Human agreement analysis using stored cache | No. |
| Final read-only committee aggregation | No live server if cache is complete. |
| Fresh remote committee run | Yes. |
| Fresh local committee collection | Yes, to the local endpoint. |
| Sharanga model serving | Cluster/GPU access required. |
| `test_api_keys.py` remote checks | Yes. |

This distinction is important for artifact review: an ACL reviewer can verify
existing results and formulas without access to the original GPU cluster.

## 10. Setup failure interpretation

- Import failure: install the root requirements in the active environment.
- Missing NLTK data: download tokenizer data, then rerun installation check.
- Missing `.env`: create it only for a provider mode that needs secrets.
- Local endpoint refusal: inspect server job, host/port, and probe output; do
  not change model ids in the evaluator to hide an endpoint mismatch.
- Cache miss in read-only mode: complete the corresponding collection stage;
  do not silently fall back to a fresh server call.
- Workbook `openpyxl` failure: repair the environment dependency path, not the
  numeric source results.

## 11. Reproduction record

For a fresh paper-facing run, record:

- Python version and environment identifier;
- dependency installation source and relevant package versions;
- input and gold file paths;
- committee config and prompt bundle version;
- endpoint model ids and served names;
- cache mode and cache root;
- output directory and run label;
- validation command result; and
- audit command result.

The record belongs beside the result artifacts or in the run's config/provenance
metadata. It should never contain API keys.

