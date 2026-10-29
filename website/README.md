# Course website

The course website is built from the repository's Markdown with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Learners do not need this folder: they can read the course on GitHub or Colab. It is here for whoever maintains the site.

## Preview it locally

From the repository root:

```bash
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r website/requirements.txt   # once
python website/build.py serve
```

Then open **http://127.0.0.1:8000/training-ai-ml/**. Opening `http://127.0.0.1:8000/` also works: it redirects there. The pages are copied when the server starts, so after you edit a lesson, stop the server with `Ctrl+C` and run the command again.

## Build the static site

```bash
python website/build.py
```

The finished site goes in `website/_site/`.

## Publishing

[`.github/workflows/website.yml`](../.github/workflows/website.yml) builds the site and publishes it to https://cloudops-one.github.io/training-ai-ml/ whenever a push to `main` changes a Markdown file or this folder. To publish by hand, open GitHub > Actions > "Website" > Run workflow. Only pages committed to git are included.

## How it works

- `build.py` copies the lessons, module intros, setup notes and cheatsheets into `_src/`, then runs MkDocs on them.
- Links to notebooks, code and data point at GitHub. GitHub shows notebooks with their outputs and has the "Open in Colab" button.
- `_src/`, `_site/` and `_mkdocs.generated.yml` are generated. Edit the original files in `modules/`, `setup/` and so on, never the copies here.
- `mkdocs.yml` holds the theme and site settings.
