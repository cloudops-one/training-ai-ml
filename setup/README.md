# Setup

You have two options. Pick one; you can switch later.

## Option A: Google Colab (recommended to start)

Nothing to install. Every `lesson.ipynb` has an "Open in Colab" badge at the top. Click it, then **Runtime → Run all**.

- Colab already has NumPy, Pandas, scikit-learn, TensorFlow and most libraries this course uses.
- Modules 08 and 09 need a GPU for a few notebooks: **Runtime → Change runtime type → T4 GPU**.
- Colab forgets files when the session ends. Each notebook re-downloads the data it needs, so that is fine.

## Option B: Local install

**Use Python 3.12.** Every library in the main course (including TensorFlow) has been checked against it. Python 3.13 and 3.14 are too new for some of them, and 3.10/3.11 work but are not what the notebooks were tested on.

The one exception is the optional PyCaret lesson (05.9): PyCaret only runs on Python 3.9–3.11, so it gets its own small environment. See [requirements-pycaret.txt](requirements-pycaret.txt). You can skip it without missing anything the rest of the course depends on.

Check what you have:

```bash
python3 --version
```

If it is not 3.12, install it alongside your current Python: on macOS `brew install python@3.12`; on Windows use the installer from python.org; on Ubuntu `sudo apt install python3.12 python3.12-venv`. Then use it explicitly below (`python3.12` instead of `python3`).

```bash
git clone https://github.com/cloudops-one/training-ai-ml.git
cd training-ai-ml
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r setup/requirements.txt
jupyter lab
```

Then open `setup/check_environment.ipynb` and run it. It prints the version of every library and tells you if anything is missing.

Deep learning (Module 08) and generative AI (Module 09) have extra, heavier dependencies. Install them when you get there:

```bash
pip install -r setup/requirements-deep-learning.txt   # Module 08
pip install -r setup/requirements-genai.txt           # Module 09
```

Apple Silicon users: TensorFlow installs fine from `pip`; a GPU is not needed for this course's deep-learning notebooks, they are sized to run on a laptop CPU in a few minutes.

## LLM access for Module 09

Lessons that call a large language model use the Anthropic API. If you have a key, put it in a `.env` file at the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is git-ignored. Never commit keys.

No key? Every LLM lesson has a **local fallback** cell that uses a small open model through [Ollama](https://ollama.com). Install Ollama, run `ollama pull llama3.2`, and the notebooks will detect it.

## Datasets

Run once:

```bash
python data/download.py
```

This fetches the Online Retail II dataset (about 45 MB) into `data/raw/`. Small example tables are already in `data/` and need nothing.

## Checking everything works

```bash
jupyter nbconvert --execute --to notebook setup/check_environment.ipynb
```

If that runs without errors, you are ready. Start with [Module 00](../modules/00-big-picture/).
