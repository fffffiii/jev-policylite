# Preview and publish the project page

The project page is a static site in [`site/`](../site). It is separate from the local moderation service and never submits images to that service.

## Local preview

There is no build step:

```bash
python -m http.server 4173 --directory site
```

Open `http://127.0.0.1:4173/`. Diagrams, stage tabs and command examples are local files. Documentation links lead to the public GitHub repository.

## Pages deployment

The included [workflow](../.github/workflows/pages.yml) uploads only `site/`. Keep private data, checkpoints and service addresses out of `site/`.

The public project page is [fffffiii.github.io/jev-policylite](https://fffffiii.github.io/jev-policylite/). Pages uses **GitHub Actions** as its source. The workflow runs on changes to `site/` on `main`, or can be run manually.

[`site/assets/config.js`](../site/assets/config.js) defines the repository URL. Update it when moving the project. Documentation links use the repository's `main` branch, so changes on a review branch are not live until that PR is merged. A documentation-review PR does not require publishing a site.
