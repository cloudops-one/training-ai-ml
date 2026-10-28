# 10.2 · A prediction API with FastAPI

⏱ about 1.5 hours · Prerequisites: 10.1 · Notebook: [lesson.ipynb](lesson.ipynb) · Program: [api.py](api.py)

---

## 1. The problem

The web team wants a small "at risk" badge on the customer-service screen. When an agent opens a customer, the website should ask the churn model *"how likely is this person to leave?"* and get an answer in well under a second.

The website is not written in Python and cannot open a joblib file. A weekly CSV (10.1) is a week too late. The model needs a **web address** that any program can call.

## 2. The idea

Wrap the model in a tiny web service. The website sends the customer's numbers as JSON to a URL; the service checks them, runs the model, and sends back JSON:

```
website  ──POST /predict {"orders": 2, "spend": 310, ...}──▶  churn API  ──▶ model
website  ◀──────── {"churn_probability": 0.52, "voucher": true} ───────────┘
```

This contract (URL, input fields, output fields) is an **API**. As long as the contract stays the same, you can swap the model behind it without the website noticing.

## 3. Definition

- **API** (application programming interface): an agreed way for programs to talk. Here, an **HTTP** API: requests to URLs, data as **JSON**.
- **Endpoint**: one URL plus method. `GET /health` asks "are you alive?"; `POST /predict` sends data to be scored.
- **FastAPI**: a Python library for building HTTP APIs from ordinary functions with type hints.
- **Pydantic model**: a class that describes the expected input, with types and limits. FastAPI uses it to **validate** every request and reject bad ones with status **422** before your code runs.
- **Uvicorn**: the server program that listens on a port and hands requests to FastAPI.
- **Status codes**: 200 OK, 422 invalid input, 500 server error.
- **Latency**: the time one request takes. Reported as percentiles: **p50** (typical) and **p95** (slow but not rare).

## 4. Formula

The **k-th percentile** of n sorted timings is (roughly) the value at position $\lceil k/100 \cdot n \rceil$:

$$p_{50} = \text{the middle timing}, \qquad p_{95} = \text{the timing only 5\% of requests exceed}$$

**Throughput** for one worker handling one request at a time:

$$\text{requests per second} \approx \frac{1}{\text{mean latency in seconds}}$$

Averages hide slow requests, which are the ones users notice. That is why services promise p95 or p99, not the mean.

## 5. By hand

Ten request timings in milliseconds, sorted: **1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.6, 1.8, 2.4, 9.0**.

- p50 = halfway between the 5th and 6th values (1.4 and 1.5) = **1.45 ms**.
- p95 = position ⌈0.95 × 10⌉ = 10th value = **9.0 ms**.
- mean = 22.9 / 10 = **2.29 ms**, pulled up by the one slow request.
- throughput ≈ 1 / 0.00229 ≈ **437 requests per second** for one worker.

One slow request in ten makes the mean look 60% worse than a typical request, and the p95 shows the real worst case.

## 6. In code

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Customer(BaseModel):
    orders: int = Field(ge=1)
    spend: float = Field(ge=0)
    ...

app = FastAPI()
bundle = load_bundle()                       # once, at start-up

@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[bundle["features"]]
    p = bundle["pipeline"].predict_proba(row)[0, 1]
    return Prediction(churn_probability=p, voucher=p >= bundle["threshold"], ...)
```

The full service is [api.py](api.py): `/health`, `/predict` for one customer, `/predict/batch` for many, and a check that `days_since_last` is not larger than `tenure_days`.

**Run it** from this folder (same command on Windows, macOS and Linux):

```bash
python -m uvicorn api:app --reload
```

Open **http://127.0.0.1:8000/docs**: FastAPI writes an interactive page from your code where you can try every endpoint. Stop the server with Ctrl+C.

**Call it** from another terminal:

macOS / Linux:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"orders": 2, "spend": 310, "products": 14, "tenure_days": 150, "days_since_last": 95}'
```

Windows (PowerShell):
```powershell
$body = @{orders=2; spend=310; products=14; tenure_days=150; days_since_last=95} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

**Test it without a server.** In a notebook, `fastapi.testclient.TestClient(app)` sends requests straight to the app. That is how the lesson notebook works, and how real teams write automated tests.

## 7. On real data

The notebook loads `api.py` and tests it:

- `/health` reports the model's training date, test AUC 0.741 and threshold 0.07.
- The 05.10 customer scores **0.5244** with a voucher, exactly as in 10.1.
- A negative spend is rejected with **422** and a message naming the field; `days_since_last` 200 with `tenure_days` 150 is rejected by the custom check. The model never sees bad data.
- **Latency**: one request through the app takes about **1.5 ms** at p50 and **1.7 ms** at p95 on the machine that produced these outputs, with the occasional slow one (the slowest of 200 took 20 ms). Scoring the **830 test customers** one request at a time takes **1.2 s**; one `/predict/batch` request with all 830 takes **4 ms**, a few hundred times faster, with identical answers.

These timings leave out the network. A real call from a browser adds anything from a few to a few hundred milliseconds, which is usually far more than the model.

## 8. Where it breaks

- **Loading the model on every request.** Reading the file each time multiplies latency. Load once at start-up.
- **No input validation.** Without the Pydantic limits, `"spend": -50` or `"orders": "two"` would reach the model and produce a confident number. Validate at the edge.
- **Changing the contract silently.** Renaming `orders` to `n_orders` breaks every caller. Add new fields, keep old ones, or version the URL (`/v2/predict`).
- **`--reload` in production.** It restarts the server whenever a file changes; useful while developing, wrong in production. Production runs several workers behind a proxy, usually in a container (10.4).
- **Open to the world.** This API has no authentication. Inside ShopWave's network that may be acceptable; on the internet it is not.
- **No record of what was asked.** Without logging requests and predictions, you cannot monitor drift later (10.6). The project's API logs every prediction.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: for the timings 3, 2, 2, 4, 2, 3, 15, 2, 3, 2 ms, compute p50, p95, the mean, and one-worker throughput.
2. Add a `/model` endpoint that returns the bundle's `data` description and metrics. Test it with `TestClient`.
3. Send 50 customers to `/predict/batch` where one row has `orders: 0`. What happens to the whole batch? Is that the behaviour you want? Propose an alternative.

**Quiz**

1. What does status code 422 mean, and who produces it in this service?
2. Why load the bundle once at start-up rather than inside `predict`?
3. Why report p95 latency rather than the mean?
4. What is the difference between `GET /health` and `POST /predict`?
5. Give two reasons to call `/predict/batch` instead of `/predict` 830 times.

## 10. Recap

- An API turns the model into a **contract**: a URL, input fields, output fields, status codes.
- FastAPI builds it from functions with type hints; **Pydantic** validates every request and returns 422 for bad input.
- Run with `python -m uvicorn api:app`; explore at `/docs`; call with curl or PowerShell's `Invoke-RestMethod`.
- Test with `TestClient`, no server needed.
- Measure **p50 and p95** latency; batch requests when you can; load the model once.

**Next:** [10.3 A web app with Streamlit](../03-streamlit/)
