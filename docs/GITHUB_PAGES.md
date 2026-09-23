# Preview and publish the project page

The project page is a static site in [`site/`](../site). It is separate from the local moderation service and never submits images to that service.

## Local preview

There is no build step:

```bash
python -m http.server 4173 --directory site
```

Open `http://127.0.0.1:4173/`. Diagrams, stage tabs and command examples are local files. Documentation links lead to the configured GitHub repository; they may require your GitHub login while that repository is private.

## Pages deployment

The included [workflow](../.github/workflows/pages.yml) uploads only `site/`. Before enabling deployment, verify the intended visibility of both the repository and the resulting site. Do not change repository visibility just to preview a page, and do not put private data, checkpoints or service addresses into `site/`.

In repository settings, configure **Pages → Source → GitHub Actions** when that option is available. The workflow runs on changes to `site/` on `main`, or can be run manually. Use the URL returned by a successful deployment; this document does not assert that the site is currently published.

[`site/assets/config.js`](../site/assets/config.js) defines the repository URL. Update it when moving the project. Documentation links use the repository's `main` branch, so changes on a review branch are not live until that PR is merged. A documentation-review PR does not require publishing a site.
