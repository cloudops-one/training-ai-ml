# 10.4 · Docker in one hour (optional)

⏱ about 1 hour · Prerequisites: 10.1, 10.2 · Notebook: [lesson.ipynb](lesson.ipynb) · Files: [Dockerfile](Dockerfile), [Dockerfile.dockerignore](Dockerfile.dockerignore), [requirements-api.txt](requirements-api.txt)

**Optional.** You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS) or Docker Engine (Linux) to build and run images. Without it, you can still read the files and run the notebook: it checks everything that does not need Docker and shows every command.

---

## 1. The problem

The API from 10.2 runs on your laptop. To run it on ShopWave's server, someone must install the right Python, the right versions of eight libraries, copy three files to the right places, set an environment variable and start uvicorn with the right options. They will get one of those wrong. And "it works on my machine" will not help them at 2am.

## 2. The idea

Ship the **whole machine setup** with the program. A Docker **image** is a frozen package: an operating system's files, Python, the pinned libraries, your code and your model, plus the command to start it. Anyone with Docker runs it with one command and gets exactly what you tested, on Windows, macOS, Linux or a cloud server.

Think of a shipping container: the crane does not care what is inside, because every container has the same shape.

## 3. Definition

- **Image**: a read-only package of files and settings, built from a recipe. Named like `shopwave-churn`.
- **Container**: a running instance of an image, isolated from your machine. Many containers can run from one image.
- **Dockerfile**: the recipe: a base image (`FROM`), files to copy in (`COPY`), commands to run while building (`RUN`), settings (`ENV`, `EXPOSE`) and the start command (`CMD`).
- **Layer**: each step's result. Docker caches layers and reuses them until a step's inputs change.
- **Build context**: the folder whose files the build can copy, sent to the builder at the start. An **ignore file** controls what is sent.
- **Port mapping**: `-p 8000:8000` connects port 8000 on your machine to port 8000 in the container.

## 4. Formula

Image size is roughly additive:

$$\text{image} \approx \text{base image} + \text{libraries} + \text{your code and model}$$

And the caching rule: when you rebuild, Docker reuses every step **before** the first step whose inputs changed, and redoes that step and **every step after it**. So put slow, rarely-changing steps (installing libraries) first, and fast, often-changing ones (your code, your model) last.

## 5. By hand

The pinned libraries, measured as installed on the machine that produced these outputs:

| library | MB |
|---|---|
| scipy | 97.6 |
| pandas | 77.0 |
| scikit-learn | 46.0 |
| numpy | 32.6 |
| pydantic, joblib, fastapi, uvicorn | 8.1 together |
| **total** | **261** |

Our code is two Python files and the model is 2 KB. So nearly all of the image is the base image plus the scientific stack, and the model is a rounding error. That is the case for most classical ML services, and the reason the library step is cached separately.

**Which steps rebuild when you retrain the model?** The steps run: base, workdir, copy requirements, install, copy `churn_model.py`, copy `api.py`, copy model, … The model copy is the first changed step, so only it and what follows rerun: seconds, not minutes.

## 6. In code

[Dockerfile](Dockerfile), the important lines:

```dockerfile
FROM python:3.12-slim                        # same Python as the course
WORKDIR /app
COPY modules/10-deployment-mlops/04-docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt      # slow, cached
COPY lib/churn_model.py lib/
COPY modules/10-deployment-mlops/02-fastapi/api.py .
COPY models/churn_bundle.joblib models/
ENV CHURN_MODEL_PATH=/app/models/churn_bundle.joblib
USER appuser                                 # not root
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

[requirements-api.txt](requirements-api.txt) pins the exact versions the model was saved with, and [Dockerfile.dockerignore](Dockerfile.dockerignore) lets only the four needed files into the build.

**Build and run**, from the course root (same on Windows, macOS and Linux):

```bash
python modules/10-deployment-mlops/01-saving-and-reproducibility/train_churn.py
docker build -f modules/10-deployment-mlops/04-docker/Dockerfile -t shopwave-churn .
docker run --rm -p 8000:8000 shopwave-churn
```

Then open http://127.0.0.1:8000/docs, or call `/health` with curl (macOS/Linux) or `Invoke-RestMethod` (PowerShell). Ctrl+C stops the container.

Windows: Docker Desktop needs WSL2 enabled (its installer offers to do it). Linux: add yourself to the `docker` group or prefix commands with `sudo`.

## 7. On real data

The notebook, without needing Docker:

- checks that every file the Dockerfile copies exists, and that the ignore file lets in exactly those four;
- measures what the ignore file saves: the course folder holds **3.5 GB** of virtual environment, about 190 MB of data and the models from every module; the build receives **13 KB**;
- measures the libraries (the table above);
- shows `CHURN_MODEL_PATH` redirecting `churn_model.py` to `/app/models/...`, the one line of the code that knows it is in a container.

If Docker is running, it then builds the image, prints its size, starts a container, calls `/health` and stops it. The outputs committed with the course were produced on a machine where Docker was not running, so that cell shows the "not running" message. The pinned versions were checked to exist as Linux packages for Python 3.12 on both Intel/AMD and ARM machines.

## 8. Where it breaks

- **Building from the wrong folder.** The `COPY` paths are relative to the build context, the course root. Run `docker build` from there with `-f` pointing at the Dockerfile.
- **No model, or an old model.** The image contains whatever `models/churn_bundle.joblib` was at build time. Retrain, then rebuild; tag images with the model version (`shopwave-churn:2026-10-28`).
- **Unpinned libraries.** `scikit-learn` without `==` installs whatever is newest on build day, which may not load your pickle.
- **Secrets in images.** Never `COPY .env` or put API keys in a Dockerfile; anyone with the image can read them. Pass them at run time (`docker run -e KEY=...`) or use a secrets manager.
- **Running as root.** If the service is broken into, the attacker has root inside the container. The `USER appuser` line prevents that.
- **Apple Silicon vs Intel servers.** An image built on an M-series Mac is ARM; most cloud servers are Intel/AMD. Build for the server with `docker build --platform linux/amd64 ...`.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Which steps rebuild when (a) you retrain the model, (b) you fix a typo in `api.py`, (c) you upgrade FastAPI?
2. Write a Dockerfile that serves the Streamlit app from 10.3 on port 8501.
3. (Needs Docker.) Build and run the API image and check `/predict` returns 0.5244 for the 05.10 customer. What happens if you build before training?

**Quiz**

1. What is the difference between an image and a container?
2. Why does the Dockerfile install the libraries before copying the code?
3. What would happen without the ignore file?
4. Why are the library versions pinned with `==`?
5. Why does the image switch to `USER appuser`?

## 10. Recap

- An **image** packages OS files, Python, pinned libraries, code and model; a **container** runs it the same way everywhere.
- Write the Dockerfile **slow-and-stable first, fast-and-changing last** so rebuilds reuse cached layers.
- Use an **ignore file**: send the builder four files, not gigabytes.
- **Pin** library versions to the ones the model was saved with; never bake secrets into an image; don't run as root.
- `docker build -f ... -t name .` then `docker run --rm -p 8000:8000 name`.

**Next:** [10.5 Experiment tracking with MLflow](../05-mlflow/)
