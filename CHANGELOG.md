# Changelog

## Unreleased — 2026-09-23

### Documentation and figures

- Replaced promotional and ambiguous wording in both READMEs, the project page, local UI, guides, model card, CLI help and model/head docstrings.
- Standardized **violation head**, **attribute head** and **policy head** while preserving historical filenames and JSON keys.
- Clarified that discrete DPO is not online RL, `chosen/rejected` are actions, the feature cache is process-local, and the 0.19 s historical measurement includes evaluation but excludes extraction/loading/saving.
- Redrew the architecture and DPO figures as editable SVGs, with regenerated PNG fallbacks and a reproducible renderer. Removed deployment promises and benchmark numbers from the diagrams.
- Corrected the mosaic interpretation: the repository does not provide a paired before/after result that isolates the effect of shrinking images.
- Documented temporary image storage, absent authentication, the limits of optional imported metrics, and outstanding fixture-image provenance.

### Documentation-related fixes

- Preserved explicit image-group identifiers when converting human feedback. Older records fall back to image paths rather than independent feedback IDs.
- Made offline test metrics optional for service startup; missing, malformed and non-finite reports no longer produce fabricated or broken metrics displays. Added a required-processor check.
- Prevented an old custom rule from silently overriding a subsequently selected preset.
- Replaced fixed sample counts, GPU assumptions and upload limits with returned values. Escaped imported display strings and handled missing scores.
- Kept policy suggestions separate from the binary threshold verdict. Corrected memory units and processing-time labels.
- Included web static assets in built wheels; narrowed model-weight exceptions and ignored generated processor/build files.

### Validation

- Added regression tests for feedback grouping, optional reports, output-label compatibility, documentation links and editable diagrams.
- Added an offline Chromium interaction-check script using explicit API fixtures; it does not perform model inference or HTTP-service validation.
- Added CPU-only source/documentation checks for pull requests. Model weights, training prompts, labels and tensor computations were not changed, except for the documented grouping fix in preference preparation.

See [the review record](docs/REPOSITORY_REVIEW_2026-09-23.md) for test scope and remaining checks. Temporary source-snapshot/transfer helpers are removed from the final branch tree.
