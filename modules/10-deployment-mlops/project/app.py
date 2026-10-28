"""ShopWave assistant: churn risk for the retention team, policy answers for support.

    python -m streamlit run app.py          # from this folder, Windows / macOS / Linux

The chatbot tab needs an LLM (see setup/README.md); the churn tab works without one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # support_bot.py next to this file
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import support_bot  # noqa: E402  (also puts lib/ on the path)
import churn_model as cm  # noqa: E402

st.set_page_config(page_title="ShopWave assistant", page_icon="🛍️")
st.title("ShopWave assistant")


@st.cache_resource
def get_bundle():
    return cm.load_bundle()


@st.cache_resource(show_spinner="Reading the policy documents...")
def get_index():
    return support_bot.build_index()


churn_tab, chat_tab = st.tabs(["Churn risk", "Support chatbot"])

with churn_tab:
    bundle = get_bundle()
    st.caption(f"Model trained {bundle['trained']} · test AUC {bundle['metrics']['test_auc']}")
    voucher_cost = st.slider("Voucher cost (£)", 1, 40, 5, key="voucher_cost")
    threshold = voucher_cost / cm.LOST_CUSTOMER_COST
    left, right = st.columns(2)
    values = [left.number_input("Orders", min_value=1, value=2, key="orders"),
              left.number_input("Total spend (£)", min_value=0.0, value=310.0, step=10.0, key="spend"),
              left.number_input("Different products", min_value=1, value=14, key="products"),
              right.number_input("Days since first order", min_value=0, value=150, key="tenure_days"),
              right.number_input("Days since last order", min_value=0, value=95, key="days_since_last")]
    if values[4] > values[3]:
        st.error("Days since last order cannot be more than days since first order.")
    else:
        p = float(bundle["pipeline"].predict_proba(pd.DataFrame([values], columns=bundle["features"]))[0, 1])
        st.metric("Churn risk", f"{p:.0%}")
        if p >= threshold:
            st.success(f"Send a £{voucher_cost} voucher (break-even risk {threshold:.0%}).")
        else:
            st.info(f"No voucher (break-even risk {threshold:.0%}).")

with chat_tab:
    ok, status = support_bot.llm_status()
    if not ok:
        st.warning("The chatbot needs a language model, and none is configured.\n\n"
                   "Free option: install Ollama and run `ollama pull llama3.2`. "
                   "Or add an API key to `.env`. See setup/README.md.")
        st.caption(status)
    else:
        st.caption(f"Answers come only from ShopWave's policy documents · {status}")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])
        question = st.chat_input("Ask about delivery, returns, vouchers...")
        if question:
            st.chat_message("user").write(question)
            reply, hits = support_bot.answer(get_index(), question)
            sources = ", ".join(sorted({h["source"] for h in hits}))
            text = f"{reply}\n\n*Searched: {sources}*"
            st.chat_message("assistant").write(text)
            st.session_state.messages += [{"role": "user", "content": question},
                                          {"role": "assistant", "content": text}]
