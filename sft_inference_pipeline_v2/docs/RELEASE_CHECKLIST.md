# Public-release checklist

The code/data/result surface is cleaned and validated, but these author-owned release decisions remain before a public archival upload:

- Choose and add a repository license. Do not infer one from this checklist.
- Add a `CITATION.cff` with the final paper title, author list, venue, DOI/arXiv identifier, and release date.
- Confirm that redistribution of every base model, dataset, prompt source, and final output is permitted; add the applicable model and data license/provenance statements.
- If exact model inference is required, publish the current LoRA adapters with immutable checksums and a model-revision manifest.
- State the answer-only abstention-only checkpoint-selection behavior from `METHOD_LIMITATIONS.md` in the paper or supplementary material.
- Provide the SLURM partition, GPU, CUDA driver, and base-model revision details used for the reported runs.

These decisions require author authority and are deliberately not guessed or fabricated by the repository cleanup.
