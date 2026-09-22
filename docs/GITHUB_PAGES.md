# GitHub Pages deployment

The project page is a static site in [`site/`](../site). The included workflow uploads exactly that folder and deploys it through GitHub Pages.

1. Create a public GitHub repository and push the project.
2. Open **Settings → Pages** and set the source to **GitHub Actions**.
3. Push to `main`, or run the **Deploy GitHub Pages** workflow manually.
4. The deployment URL will be shown in the workflow summary. For this repository, the default address is `https://fffffiii.github.io/jev-policylite/`.

[`site/assets/config.js`](../site/assets/config.js) already points to the public repository. Update it only if the project moves to a different repository or custom domain.

The site has no build dependency. Local preview:

```bash
python -m http.server 4173 --directory site
```

Then open `http://127.0.0.1:4173/`. Do not point the public project page at a local moderation service or include private checkpoints and data.
