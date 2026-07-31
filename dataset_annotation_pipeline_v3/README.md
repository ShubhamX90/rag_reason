# Conflict-Aware RAG Dataset Annotation Pipeline

This repository contains the released data and the documented construction workflow for a conflict-aware, evidence-grounded RAG dataset. Each example asks a system to interpret a *set* of retrieved snippets: identify useful evidence, characterize agreement or conflict, decide whether the evidence is sufficient, and produce a grounded answer or abstention.

The retained method is **stagewise**. It uses a weighted multi-LLM committee to make auditable decisions at the document, evidence-set, and final-response levels. Older one-shot/monolithic experiments are preserved under [legacies/](legacies/) but are not part of the active method or a reviewer-facing release.

## Start here: canonical release files

[data/releases/](data/releases/README.md) is the clean, reviewer-facing release surface. It contains the only dataset files that should be used for reported release counts or included in a minimal archival copy.

| Deliverable | Canonical file | Records | Intended use |
|---|---|---:|---|
| Training split | [train.jsonl](data/releases/training_dataset_v2/train.jsonl) | 862 | Model training |
| Validation split | [val.jsonl](data/releases/training_dataset_v2/val.jsonl) | 81 | Model selection and validation |
| Benchmark holdout | [benchmark_final_v2_holdout_clean_736.jsonl](data/releases/benchmark_dataset_v2/benchmark_final_v2_holdout_clean_736.jsonl) | 736 | Primary reviewer-facing evaluation holdout |
| Broader benchmark-v2 pack | [benchmark_final_v2.jsonl](data/releases/benchmark_dataset_v2/benchmark_final_v2.jsonl) | 933 | Release-pack context; not a replacement for the 736-record holdout |

The stagewise training files are variants of the same released 862/81 split. The 736-record holdout has no identifier overlap with the training or validation release.

### Do not confuse releases with retained construction artifacts

The repository intentionally preserves a small number of internal artifacts needed to understand or reproduce the method. They are useful, but they are not additional public datasets or competing “latest” versions:

| Artifact | Meaning | Not a substitute for |
|---|---|---|
| data/final_annotations/stagewise_multi/ | Historical 658-record stagewise annotation output | The final 943-record training/validation release |
| data/splits/92p5_7p5/ | Retained 609/49 internal validation split | The canonical 862/81 release split |
| data/benchmarks/final_benchmark_2026-06-22/ | 1,000-record internal benchmark-build artifact | The 736-record benchmark holdout |
| outputs/ | Retained latest committee-run outputs and diagnostics | Canonical release datasets |
| legacies/ | Recoverable superseded code, data, pilots, and exports | The active stagewise workflow |

When preparing a clean submission archive, retain the release directory and the active documentation/workflow; omit legacies/ if historical material is not required by the venue.

## What the data supervises

The task is deliberately more demanding than ordinary answer generation. A model should learn to inspect retrieved evidence rather than silently rely on outside knowledge.

1. **Document evidence.** Each snippet is judged as supports, partially supports, or irrelevant, with a grounded key fact and quote.
2. **Evidence-set structure.** The record is assigned one conflict type: No conflict, Complementary information, Conflicting opinions or research outcomes, Conflict due to outdated information, or Conflict due to misinformation.
3. **Evidence sufficiency.** The answerable_under_evidence field captures whether the retrieved material justifies a response.
4. **Response behavior.** The expected_response.abstain field distinguishes a grounded answer from a refusal, with a coherent final answer/evidence bundle when answering is justified.

This structure supports research on retrieval quality, evidence use, conflict-aware synthesis, temporal reasoning, misinformation handling, and calibrated abstention.

## End-to-end workflow

~~~text
Source queries and records
       |
       +-- Benchmark non-refusal queries: source pools -> Tavily retrieval
       +-- Refusal examples: held-out TRUST-ALIGN material (not Tavily retrieval)
       +-- Training: CONFLICTS + TRUST-ALIGN + non-holdout benchmark material
       v
Normalization and schema validation
       v
Stage 1: per-document evidence adjudication
       v
Stage 2: conflict reasoning + answerability
       v
Stage 3: grounded response or abstention
       v
Human review, consolidation, and quality selection
       v
Canonical training/validation and benchmark releases
~~~

### Annotation stages and committee method

The annotation pipeline votes on a small number of decision fields, then retains the complete explanation bundle from the highest-weight committee member on the winning side. It therefore does not average prose from different models into an incoherent annotation.

| Stage | Main decision | Result |
|---|---|---|
| 1 | Per-document evidence verdict | Evidence note for every retrieved snippet |
| 2 | Conflict type where applicable and answerability | Set-level conflict explanation and sufficiency decision |
| 3 | Abstain vs. answer | One grounded final-response bundle |

The current default OpenRouter committee is Claude Haiku 4.5 (0.35), GPT-5.4 (0.30), DeepSeek V3.2 (0.20), and Mistral Small 2603 (0.15). Historical and local construction artifacts used run-specific committees; those are reported separately rather than retroactively relabeled as the current default. See the [annotation-pipeline methodology](docs/ANNOTATION_PIPELINE.md) for all model configurations, weights, decision rules, modes, audit fields, and reproduction boundaries.

## Dataset provenance at a glance

### Benchmark

The reviewer-facing benchmark holdout contains 736 records: 608 answerable non-refusal examples and 128 refusal/insufficient-evidence examples. The broader benchmark-v2 pack contains 933 records (805 answerable and 128 refusal). Its five-label distribution is reported in full in the benchmark dataset card.

For the non-refusal track, a 2,000-query candidate pool was drawn from ConflictingQA and four source families of the CONFLICTS-style collection (SituatedQA geographic/temporal, FreshQA, and QACC). After source exclusion and deduplication, Tavily retrieval produced the candidates considered by the subsequent evidence and review workflow. The retrieval workflow began from 20,330 raw candidate records, removed 598 excluded-source rows and 8,891 duplicates, and retained 10,841 unique usable candidates before the deterministic 2,000-query collection subset.

The benchmark preselection process consolidated 1,454 records. Quality selection produced the 736-record held-out evaluation set. The 128 refusal examples are held-out TRUST-ALIGN records, not web-retrieval examples.

For exact source distributions, retrieval gates, human-review provenance, normalization, schemas, and the distinction between the 933-record pack and 736-record holdout, read:

- [Benchmark dataset description](docs/BENCHMARK_DATASET_DESCRIPTION.md)
- [Tavily retrieval methodology](docs/TAVILY_RETRIEVAL_METHODOLOGY.md)
- [Benchmark construction guide](docs/conflicts_benchmark_build.md)

### Training and validation

The final training release has 943 records: 862 training and 81 validation. It combines 458 normalized CONFLICTS records, 272 held-out TRUST-ALIGN refusal records, and 213 benchmark-source additions that are all outside the final 736-record benchmark holdout. The split contains 649 answerable and 294 non-answerable records; it has 6,642 retrieved documents (mean 7.043 per record).

The final label distribution is 343 No conflict, 286 Complementary information, 158 Conflicting opinions or research outcomes, 136 Conflict due to outdated information, and 20 Conflict due to misinformation.

The [training dataset description](docs/TRAINING_DATASET_DESCRIPTION.md) provides source-family accounting, normalization history, split integrity, human-review coverage, selection decisions, schema, and limitations.

## Human review and agreement reporting

Training conflict-type review and benchmark preselection are intentionally separate workflows, with separate raw-review and consolidated artifact trees:

| Workflow | Raw reviews | Consolidated output |
|---|---|---|
| Training conflict-type review | [human_reviews/training/reviews/](human_reviews/training/reviews/) | [training_conflict_type_consensus_943.jsonl](human_reviews/training/consolidated/training_conflict_type_consensus_943.jsonl) |
| Benchmark preselection | [human_reviews/benchmark/](human_reviews/benchmark/) | [benchmark_preselection_consensus_1454.jsonl](human_reviews/benchmark/consolidated/benchmark_preselection_consensus_1454.jsonl) |

The benchmark consolidation covers the 1,454-record preselection/review population from which the 736-record holdout was selected. The training consolidation covers the 943-record release population, including explicit provenance for records outside the recorded 658-record interactive review population.

The human-review methodology explains reviewer assignment, first/second-pass design, reconstruction metadata, consensus-completed sides, refusal-quality assessment, raw agreement, Cohen’s kappa, the formulas used, and the limits on interpreting these statistics as independent blind inter-annotator agreement:

- [Human-review methodology and agreement analysis](docs/HUMAN_REVIEW_METHODOLOGY.md)
- [Training review report](human_reviews/training/consolidated/training_conflict_type_agreement_report.md)
- [Benchmark review report](human_reviews/benchmark/consolidated/benchmark_preselection_agreement_report.md)

## Documentation map

Use the following documents as the primary research record. The top-level README gives orientation; the linked documents give the detailed, citable methodology and artifact lineage.

| If you need to know… | Read |
|---|---|
| What the canonical files are and which should ship | [Release README](data/releases/README.md) |
| Benchmark sources, statistics, selection, refusal track, and integrity | [Benchmark dataset description](docs/BENCHMARK_DATASET_DESCRIPTION.md) |
| Search, fetch, text extraction, and windowing details for benchmark evidence | [Tavily retrieval methodology](docs/TAVILY_RETRIEVAL_METHODOLOGY.md) |
| Training sources, distributions, normalization, and split construction | [Training dataset description](docs/TRAINING_DATASET_DESCRIPTION.md) |
| Exact JSONL fields, schema differences, audit metadata, and safe loading | [Data format and schema reference](docs/DATA_FORMAT_AND_SCHEMA_REFERENCE.md) |
| Models, weights, stagewise logic, prompts, modes, and run provenance | [Annotation pipeline](docs/ANNOTATION_PIPELINE.md) |
| Compact committee decision explanation | [Multi-LLM committee logic](MULTI_LLM_COMMITTEE_LOGIC.md) |
| Human-review protocol, consolidation, agreement metrics, and caveats | [Human-review methodology](docs/HUMAN_REVIEW_METHODOLOGY.md) |
| Operational benchmark reconstruction sequence | [Benchmark construction guide](docs/conflicts_benchmark_build.md) |

## Repository layout and active workflow surfaces

~~~text
data/releases/          Canonical reviewer-facing dataset files
data/final_annotations/ Historical retained stagewise annotation output
data/splits/            Retained internal split and validation artifacts
data/benchmarks/        Retained benchmark-construction artifacts
human_reviews/training/ Training review assignments, raw files, consolidation
human_reviews/benchmark/Benchmark review assignments, raw files, consolidation
src/                    Active committee, client, parser, voting, and utility code
scripts/                Active normalization, retrieval, annotation, validation,
                        benchmark-build, and review-consolidation workflows
prompts/                Stage- and mode-specific prompt templates
configs/local_committee/Local committee configurations and notes
slurm/                  Retained local/HPC orchestration material
outputs/                Retained latest committee-run outputs and diagnostics
legacies/               Superseded material kept recoverably outside active scope
docs/                   Dataset cards and methodology documents
~~~

For operational work, begin from the documented stagewise sequence rather than running a script in isolation:

1. Normalize or prepare records with the schema required by the relevant mode.
2. Run Stage 1, validate/merge the committee collection, then run Stage 2.
3. Run Stage 3 only for the applicable workflow, retaining model-vote metadata.
4. Validate structured outputs before constructing a downstream subset.
5. Keep human-review source files and consolidation outputs separate from automated annotation artifacts.

The detailed [annotation-pipeline](docs/ANNOTATION_PIPELINE.md) and [benchmark-construction](docs/conflicts_benchmark_build.md) documents specify the applicable input modes, safeguards, and order of operations. The retained shell and Slurm launchers are operational aids, not an alternative annotation method.

## Installation and reproducibility levels

The core Python workflow targets Python 3.9 or later.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Install requirements_retrieval.txt only for the optional CONFLICTS-style retrieval workflow; it is not required to inspect or consume the released JSONL files.

Three reproducibility claims should be kept distinct:

| Level | What can be reproduced | Boundary |
|---|---|---|
| Direct inspection | Released JSONL data, consolidated review records, metrics, and documented distributions | No API access needed |
| Methodological rerun | Stagewise annotation or retrieval procedure using current prompts/configuration | Model APIs, search index, and fetched web pages are time-varying |
| Byte-level replay | Exact historical generations and retrieved text | Requires the original model outputs and retrieval caches; a fresh run is not expected to reproduce them byte-for-byte |

Before reporting a new run, record the precise input file, mode, prompts, committee configuration and weights, model identifiers, date, seed where applicable, validation result, and whether output is a new experiment or a canonical release artifact.

## Reporting guidance

- Cite release counts from data/releases/, not from internal construction artifacts.
- Treat the 736-record benchmark file as the primary held-out evaluation set; do not merge it with training material or use the 933-record pack as though it were the same holdout.
- Report the provenance-specific committee rather than applying the current default committee table to historical outputs.
- Report raw agreement and chance-corrected agreement together, and describe the review design accurately. The human-review document contains the required definitions and formulas.
- Do not treat current web retrieval as a time-invariant source of the original snippets. Use retained artifacts for exact inspection and a rerun for a new methodological collection.

## License, citations, and external sources

This repository preserves source provenance in record metadata and in the dataset cards. It incorporates source-derived and held-out material from the CONFLICTS-style collections and TRUST-ALIGN, alongside the repository’s own retrieval, normalization, annotation, and review artifacts. Consult the relevant source papers and their licenses before redistributing source-derived material or making claims beyond the documented scope.

For paper-writing, the dataset cards and methodology documents above are the authoritative source of counts, source lineage, review interpretation, and reproducibility limits.
