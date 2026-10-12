# Release checklist

- [x] Provide public Hub destinations and local checkpoint alternatives.
- [ ] Add the public paper URL and final BibTeX entry.
- [ ] Upload `RelMLLM-4B` and `RelMLLM-8B` as separate model repositories.
- [ ] Upload this directory as the inference-code repository.
- [ ] Verify both model repositories contain `README.md`, `LICENSE`, and `NOTICE`.
- [ ] Verify uploaded model files against the selected release version.
- [ ] Run one grounding and one VQA example from the public repositories.
- [ ] Confirm that no dataset files, annotations, credentials, or private paths are present.

## Audit Status (2026-10-12)

- [x] Author-approved recovered checkpoint version selected; no diagnostic adapters.
- [x] Original complete model files verified for both sizes.
- [x] Complete four-dataset recovered metrics compared with paper, discrepancies disclosed.
- [x] Five CPU unit tests passed in the recovered runtime.
- [x] Official dataset access links and custom selector/cache availability gaps documented.
- [x] Private source defaults removed from evaluation scripts; no credentials embedded.
- [ ] Hugging Face upload revisions and verification recorded.

No fresh GPU inference or end-to-end public download smoke is claimed here.
Training and exact custom dataset reconstruction remain outside this initial release.
