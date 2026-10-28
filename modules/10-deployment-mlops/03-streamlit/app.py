"""ShopWave churn checker: a small web app for the retention team.

Start it from this folder, with the course environment active:

    python -m streamlit run app.py               # Windows, macOS and Linux

Your browser opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.
"""
import sys
from pathlib import Path


def find_lib() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "lib" / "churn_model.py").exists():
            return folder / "lib"
    sys.exit("Cannot find lib/churn_model.py. Run this from inside the course folder.")


sys.path.insert(0, str(find_lib()))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from churn_model import load_bundle  # noqa: E402

st.set_page_config(page_title="ShopWave churn checker", page_icon="🛍️")


@st.cache_resource            # load the model once, not on every click
def get_bundle():
    return load_bundle()


bundle = get_bundle()
scaler, model = bundle["pipeline"][0], bundle["pipeline"][-1]

st.title("ShopWave churn checker")
st.caption(f"Model trained {bundle['trained']} · test AUC {bundle['metrics']['test_auc']}")

with st.sidebar:
    st.header("Campaign costs")
    voucher_cost = st.slider("Voucher cost (£)", 1, 40, 5, key="voucher_cost")
    lost_cost = st.slider("Margin lost when a customer churns (£)", 10, 200, 60, key="lost_cost")
    threshold = voucher_cost / lost_cost
    st.write(f"Send a voucher when churn risk is above **{threshold:.1%}** (voucher ÷ lost margin).")

st.subheader("Customer history")
left, right = st.columns(2)
orders = left.number_input("Orders", min_value=1, value=2, key="orders")
spend = left.number_input("Total spend (£)", min_value=0.0, value=310.0, step=10.0, key="spend")
products = left.number_input("Different products", min_value=1, value=14, key="products")
tenure = right.number_input("Days since first order", min_value=0, value=150, key="tenure_days")
recency = right.number_input("Days since last order", min_value=0, value=95, key="days_since_last")

if recency > tenure:
    st.error("Days since last order cannot be more than days since first order.")
    st.stop()

row = pd.DataFrame([[orders, spend, products, tenure, recency]], columns=bundle["features"])
p = float(bundle["pipeline"].predict_proba(row)[0, 1])

st.subheader("Result")
st.metric("Churn risk", f"{p:.0%}")
if p >= threshold:
    st.success(f"Send a £{voucher_cost} voucher: expected loss avoided £{p * lost_cost:.2f} is more than its cost.")
else:
    st.info(f"No voucher: expected loss £{p * lost_cost:.2f} is less than the £{voucher_cost} voucher.")

# Why? Each feature's push on the log-odds: coefficient × standardised value (lesson 10.3).
z = (row.to_numpy()[0] - scaler.mean_) / scaler.scale_
push = pd.Series(model.coef_[0] * z, index=bundle["features"]).sort_values()
st.subheader("What drives this score")
st.bar_chart(push, horizontal=True)
st.caption("Bars to the right raise the churn risk, bars to the left lower it, compared with an average customer.")
st.write(f"Strongest factor: **{push.abs().idxmax()}** ({'raises' if push[push.abs().idxmax()] > 0 else 'lowers'} the risk).")
