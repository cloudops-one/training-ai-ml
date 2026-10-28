"""Build the course website from the repository.

    pip install -r website/requirements.txt
    python website/build.py            # builds website/_site
    python website/build.py serve      # live preview at http://127.0.0.1:8000

The Markdown pages (lessons, module intros, setup, cheatsheets) are copied
into website/_src and rendered by MkDocs. Links to notebooks and code files
are pointed at GitHub, which shows notebooks with their outputs and has the
"Open in Colab" button; learners run notebooks in Jupyter or Colab anyway.
Nothing in the course itself is changed.
"""
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SRC = HERE / "_src"
SKIP = (".github/", "website/")
GITHUB = "https://github.com/cloudops-one/training-ai-ml/blob/main/"
LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s]+)\)")


def to_github(match, page: str, pages: set) -> str:
    """Keep links to pages; point folder links at the folder's README so MkDocs can
    fix their depth; send everything else (notebooks, code, data) to GitHub."""
    target = match.group(1)
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    anchor = "#" + anchor if anchor else ""
    if resolved in pages:
        return match.group(0)
    if posixpath.join(resolved, "README.md") in pages:
        return f"]({path.rstrip('/')}/README.md{anchor})"
    return f"]({GITHUB}{resolved}{anchor})"

MATHJAX_CONFIG = """window.MathJax = {
  tex: {inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]],
        processEscapes: true, processEnvironments: true},
  options: {ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex"}
};
document$.subscribe(() => { MathJax.typesetPromise() });
"""


def copy_course() -> int:
    if SRC.exists():
        shutil.rmtree(SRC)
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    pages = {name for name in files if name.endswith(".md") and not name.startswith(SKIP)}
    for name in pages:
        target = SRC / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / name).read_text(encoding="utf-8")
        target.write_text(LINK.sub(lambda m: to_github(m, name, pages), text), encoding="utf-8")
    (SRC / "javascripts").mkdir(exist_ok=True)
    (SRC / "javascripts" / "mathjax.js").write_text(MATHJAX_CONFIG)
    return len(pages)


def title(page: str) -> str:
    """The page's first '# ' heading, which is how the course names every page."""
    for line in (SRC / page).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace('"', "'")
    return page


def write_nav() -> Path:
    """Write a config with an explicit menu: modules in order, lessons by title."""
    def entry(page, indent):
        return f'{" " * indent}- "{title(page)}": {page}'

    lines = ["INHERIT: mkdocs.yml", "nav:", '  - "Home": README.md',
             '  - "Setup": setup/README.md', '  - "Roadmap": ROADMAP.md', '  - "Modules":']
    for module in sorted(p for p in (SRC / "modules").iterdir() if p.is_dir()):
        intro = f"modules/{module.name}/README.md"
        lines.append(f'    - "{title(intro).replace("Module ", "")}":')
        lines.append(f"      - {intro}")
        for readme in sorted(module.glob("*/**/README.md")):
            page = readme.relative_to(SRC).as_posix()
            if "capstone_template" not in page:
                lines.append(entry(page, 6))
    lines.append('  - "Cheatsheets":')
    lines.append("    - cheatsheets/README.md")
    lines += [entry(p.relative_to(SRC).as_posix(), 4)
              for p in sorted((SRC / "cheatsheets").glob("*.md")) if p.name != "README.md"]
    lines += ['  - "Glossary": GLOSSARY.md', '  - "References": REFERENCES.md',
              '  - "Data":', "    - data/README.md"]
    lines += [entry(p.relative_to(SRC).as_posix(), 4)
              for p in sorted((SRC / "data" / "shopwave_docs").glob("*.md"))]
    config = HERE / "_mkdocs.generated.yml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


if __name__ == "__main__":
    print(f"copied {copy_course()} pages into {SRC.relative_to(ROOT)}")
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    subprocess.run([sys.executable, "-m", "mkdocs", command, "-f", str(write_nav())], check=True)
