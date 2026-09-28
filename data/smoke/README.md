# Smoke-test fixtures

The four JPEGs and two manifests in this directory are small development fixtures. They exercise image loading, batching and the training loop; they are not a moderation benchmark.

The manifests include artificial labels for pipeline checks. In particular, the labels in `train-manifest.jsonl` must not be interpreted as human moderation judgments about the photographed subjects. Do not train a production classifier or report accuracy claims from these fixtures.

The four original photographs remain as smoke-test fixtures. The repository does not document their sources or individual licenses. Their reuse rights are unverified; the MIT license for the code should not be read as permission to reuse these photographs. The manifests are unchanged.
