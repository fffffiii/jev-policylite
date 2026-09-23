# Smoke-test fixtures

The four JPEGs and two manifests in this directory are small development fixtures. They exercise image loading, batching and the training loop; they are not a moderation benchmark.

The manifests include artificial labels for pipeline checks. In particular, the labels in `train-manifest.jsonl` must not be interpreted as human moderation judgments about the photographed subjects. Do not train a production classifier or report accuracy claims from these fixtures.

The repository snapshot does not include per-image source and licensing records for these four photographs. Those records should be verified and added before redistribution. The documentation review leaves the images and existing manifests unchanged rather than redrawing photographs or silently changing test data.
