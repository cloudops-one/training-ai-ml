# Setup

The course runs on **Windows, macOS and Linux**, or in **Google Colab** with nothing to install. Pick one; you can switch later.

## Option A: Google Colab (easiest start)

Every `lesson.ipynb` has an "Open in Colab" badge at the top. Click it, then **Runtime → Run all**.

- Colab already has NumPy, Pandas, scikit-learn, TensorFlow and most libraries this course uses.
- Modules 08 and 09 run faster with a GPU: **Runtime → Change runtime type → T4 GPU**.
- Colab forgets files when the session ends. Each notebook re-downloads the data it needs, so that is fine.

## Option B: Your own computer

### 1. Install Python 3.12

Every library in the main course has been checked against **Python 3.12**. Python 3.13 and 3.14 are too new for some of them (TensorFlow among them).

| | How |
|---|---|
| **Windows** | Download the 3.12 installer from [python.org](https://www.python.org/downloads/windows/). Tick **"Add python.exe to PATH"**. Check with `py -3.12 --version`. |
| **macOS** | `brew install python@3.12` (or the installer from python.org). Check with `python3.12 --version`. |
| **Linux** | Ubuntu 24.04 and newer: `sudo apt install python3.12 python3.12-venv`. Ubuntu 22.04: add the deadsnakes PPA first (`sudo add-apt-repository ppa:deadsnakes/ppa`). Fedora: `sudo dnf install python3.12`. |

The optional PyCaret lesson (05.9) is the one exception: PyCaret only runs on Python 3.9–3.11, so it gets its own small environment. See [requirements-pycaret.txt](requirements-pycaret.txt). Skipping it costs you nothing later.

### 2. Get the course and create an environment

**macOS / Linux**

```bash
git clone https://github.com/cloudops-one/training-ai-ml.git
cd training-ai-ml
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r setup/requirements.txt
jupyter lab
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/cloudops-one/training-ai-ml.git
cd training-ai-ml
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r setup/requirements.txt
jupyter lab
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In the old Command Prompt use `.venv\Scripts\activate.bat` instead.

VS Code works too: open the folder, then pick the `.venv` interpreter as the notebook kernel.

### 3. Check everything works

Open `setup/check_environment.ipynb` and run it. It prints the version of every library and tells you what is missing.

### 4. Extra libraries, when you reach them

```bash
pip install -r setup/requirements-deep-learning.txt   # before Module 08
pip install -r setup/requirements-genai.txt           # before Module 09
```

Deep learning on each system:

- **macOS (Apple Silicon)**: TensorFlow and PyTorch install from `pip`. PyTorch uses the Mac's GPU automatically.
- **Windows**: TensorFlow runs on the CPU only on native Windows (its GPU support needs WSL2). PyTorch supports NVIDIA GPUs directly. The Module 08 notebooks are sized to finish on a laptop CPU in a few minutes, so a GPU is optional.
- **Linux**: both use an NVIDIA GPU if CUDA drivers are installed, otherwise the CPU.

## Datasets

```bash
python data/download.py
```

This fetches the Online Retail II dataset (about 45 MB) into `data/raw/` and makes a fast copy for the notebooks. Small example tables are already in `data/`.

---

## Large language models for Module 09

Module 09 notebooks talk to an LLM through one small helper, [`lib/course_llm.py`](../lib/course_llm.py). You choose the provider; the notebooks do not change.

| Provider | Cost | Needs | Set in `.env` |
|---|---|---|---|
| **Ollama** (local) | Free | Ollama installed, a model pulled | `LLM_PROVIDER=ollama` |
| **Anthropic** (Claude) | Paid per use | API key from console.anthropic.com | `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` |
| **OpenAI** | Paid per use | API key | `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` |
| **Google Gemini** | Free tier | API key from aistudio.google.com | `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` |
| **Groq** | Free tier | API key from console.groq.com | `LLM_PROVIDER=groq` and `GROQ_API_KEY=...` |
| **LM Studio** (local) | Free | LM Studio's local server running | `LLM_PROVIDER=lmstudio` and `LLM_MODEL=<model name>` |

Copy [`.env.example`](../.env.example) to `.env` at the repo root and fill in your choice. `.env` is git-ignored: **never commit an API key, never paste one into a notebook.** Add `LLM_MODEL=...` to use a model other than the provider's default.

Then check your choice in any notebook:

```python
import sys; sys.path.insert(0, "lib")    # adjust the path from inside a module folder
from course_llm import self_test
self_test()
```

It reports which model is in use and whether it handles the structured output Module 09 relies on.

Embeddings (turning text into numbers for search and RAG) always run locally with `sentence-transformers`, whatever you pick above. They are free and need no key.

### Running a model on your own computer with Ollama

**Install Ollama**

| | How |
|---|---|
| **Windows** | Download and run the installer from [ollama.com/download](https://ollama.com/download). It runs in the background afterwards. |
| **macOS** | Download the app from [ollama.com/download](https://ollama.com/download), or `brew install ollama`. Start the app once. |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh`. It installs a background service. |

**Choose a model for your memory**

Your model has to share memory with your operating system, browser, Jupyter and the data in your notebooks. Pick from the row that matches your computer:

| Your memory | Pull this | Uses about | Notes |
|---|---|---|---|
| **8 GB** | `ollama pull llama3.2` | 3.4 GB | The course default. Good answers, reliable structured output. Close other heavy apps while you work. |
| 8 GB, older or struggling | `ollama pull llama3.2:1b` | 2.2 GB | Smaller and faster, noticeably less careful. Then set `LLM_MODEL=llama3.2:1b`. |
| **16 GB** | `ollama pull qwen2.5:7b` | 5.3 GB | Clearly better answers, about half the speed of `llama3.2`. Set `LLM_MODEL=qwen2.5:7b`. |
| **24 GB or more** | Your choice | | Anything up to about half your memory. Larger models write better but are slower. Run `self_test()` before you rely on one: in our tests the 20B reasoning model `gpt-oss:20b` (14 GB) got structured output right only half the time, while every small model above passed. |

Measured with the course's default 8,192-token context window, on Ollama 0.17. The "uses about" figure is model plus context; a much longer window takes much more memory (the same 3B model used 23 GB with a 131,072-token window), which is why the helper sets it for you. Change it with `LLM_CONTEXT=...` in `.env` if a lesson ever needs more.

**Graphics cards.** On Apple Silicon Macs the model shares the normal memory, so the table above is all you need. On Windows and Linux, a model that fits in an NVIDIA graphics card's own memory (VRAM) runs fast; if it does not fit, Ollama runs it on the CPU from normal memory instead. CPU-only is several times slower but fine for this course, because the answers are short.

## Checking everything works

```bash
jupyter nbconvert --execute --to notebook setup/check_environment.ipynb
```

If that runs without errors, you are ready. Start with [Module 00](../modules/00-big-picture/).
