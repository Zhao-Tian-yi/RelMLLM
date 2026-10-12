# Release checklist

- [x] Provide public Hub destinations and local checkpoint alternatives.
- [ ] Add the public paper URL and final BibTeX entry.
- [x] Upload `RelMLLM-4B` and `RelMLLM-8B` as separate model repositories.
- [x] Upload this directory as the inference-code repository.
- [x] Verify both model repositories contain `README.md`, `LICENSE`, and `NOTICE`.
- [x] Verify uploaded model files against the selected release version.
- [ ] Run one grounding and one VQA example from the public repositories.
- [x] Confirm that no dataset files, annotations, credentials, or private paths are present.

## Audit Status (2026-10-12)

- [x] Author-approved recovered checkpoint version selected; no diagnostic adapters.
- [x] Original complete model files verified for both sizes.
- [x] Complete four-dataset recovered metrics compared with paper, discrepancies disclosed.
- [x] Six CPU unit tests passed in the recovered runtime.
- [x] Official dataset access links and custom selector/cache availability gaps documented.
- [x] Private source defaults removed from evaluation scripts; no credentials embedded.
- [x] Both models' complete archived predictions independently rescored with matching results.
- [x] Public RelMLLM-4B configuration, tokenizer and processor loaded successfully.
- [x] Hugging Face upload revisions and verification recorded in reports/public_model_release.json.
- [x] Public configuration, tokenizer and processor loading passed for both models.

No fresh GPU inference or end-to-end public download smoke is claimed here.
Training and exact custom dataset reconstruction remain outside this initial release.
