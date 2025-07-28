# ---------------------------------------------------------------------------
# pre_market_bias_app.py
# ---------------------------------------------------------------------------
# PURPOSE:
#   Streamlit web app that generates a concise **Pre-Market Bias** briefing
#   for ES (S&P 500) and NQ (Nasdaq) futures.
#
#   • Pulls overnight price action via yfinance
#   • Grabs macro tickers (VIX, DXY, 10-yr yield)
#   • Fetches top business headlines with NewsAPI
#   • Feeds context to OpenAI and returns a bias summary
#
# SETUP:
#   1. `pip install -r requirements.txt`  (streamlit, yfinance, openai, requests, python-dotenv, pytz)
#   2. Add keys in `.streamlit/secrets.toml`
#        OPENAI_API_KEY = "sk-..."
#        NEWSAPI_KEY    = "..."
#      – OR – paste them into the sidebar each run.
#   3. Run locally:  `streamlit run pre_market_bias_app.py`
#   4. Deploy free on Streamlit Cloud by connecting the repo.
# ---------------------------------------------------------------------------

import os
from datetime import datetime, timedelta

import pytz
import requests
import streamlit as st
import yfinance as yf
import openai

# ---------------------------------------------------------------------------
# Helper: pull overnight stats and handle empty downloads gracefully
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # cache 5 min
def get_price_stats(ticker: str, session_tz: str = "US/Eastern") -> dict:
    """
    Return {high, low, last, prev_close, pct_change} for a futures / index symbol.
    Overnight session = 18:00 previous day → 09:30 current day ET.
    Falls back safely if Yahoo returns no data (weekends / holidays).
    """
    today = datetime.now(pytz.timezone(session_tz)).date()
    start = today - timedelta(days=1)

    intraday = yf.download(ticker, start=start, interval="5m", progress=False)
    if intraday.empty:
        return {}

    overnight = intraday.between_time("18:00", "09:30")
    high = overnight["High"].max()
    low = overnight["Low"].min()
    last = intraday["Close"].iloc[-1]

    # fetch two daily bars to ensure at least one previous close
    daily = yf.download(
        ticker,
        start=start - timedelta(days=2),
        end=start,
        interval="1d",
        progress=False,
    )
    prev_close = float(daily["Close"].iloc[-1]) if not daily.empty else last
    pct = round((last - prev_close) / prev_close * 100, 2) if prev_close else 0

    return {
        "high": round(high, 2),
        "low": round(low, 2),
        "last": round(last, 2),
        "prev_close": round(prev_close, 2),
        "pct_change": pct,
    }


# ---------------------------------------------------------------------------
# Helper: fetch news headlines (returns list[str])
# ---------------------------------------------------------------------------
def fetch_news(api_key: str, query: str = "S&P 500 OR Nasdaq futures", max_headlines: int = 5):
    url = (
        "https://newsapi.org/v2/top-headlines?"
        f"q={requests.utils.quote(query)}&language=en&category=business"
        f"&pageSize={max_headlines}&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return [f"{a['title']} — {a['source']['name']}" for a in r.json().get("articles", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helper: ask OpenAI for bias summary (expects openai.api_key already set)
# ---------------------------------------------------------------------------
def summarize_with_openai(context: str, model: str = "gpt-3.5-turbo") -> str:
    if not openai.api_key:
        return "⚠️ **OPENAI_API_KEY** is missing."
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced futures trader. "
                        "Return a concise pre-market bias for ES and NQ with key levels."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=220,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"


# ===========================================================================
# Streamlit UI
# ===========================================================================
st.set_page_config(page_title="Pre-Market Bias Generator", page_icon="⚡")
st.title("⚡ Pre-Market Bias Generator (ES / NQ)")
st.markdown("Generate a structured morning briefing in **one click**.")

with st.sidebar:
    st.header("Configuration")
    openai_key_in = st.text_input("OpenAI API Key", type="password")
    news_key_in = st.text_input("NewsAPI Key (optional)", type="password")
    run_btn = st.button("Run Generator", type="primary")

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if run_btn:
    # choose keys: sidebar ➜ secrets.toml ➜ env vars ➜ ""
    openai_key = openai_key_in or st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    news_key = news_key_in or st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))

    # set OpenAI globally
    openai.api_key = openai_key

    # ----- fetch data -----
    es = get_price_stats("ES=F")
    nq = get_price_stats("NQ=F")
    vix = get_price_stats("^VIX")
    dxy = get_price_stats("DX-Y.NYB")
    tnx = get_price_stats("^TNX")

    headlines = fetch_news(news_key) if news_key else []

    # ----- build GPT prompt -----
    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    context_lines = [
        f"Date: {now}",
        "",
        "Overnight Futures:",
        f"  • ES high {es.get('high')} / low {es.get('low')} / last {es.get('last')} ({es.get('pct_change')}%)",
        f"  • NQ high {nq.get('high')} / low {nq.get('low')} / last {nq.get('last')} ({nq.get('pct_change')}%)",
        "",
        "Macro:",
        f"  • VIX {vix.get('last')} ({vix.get('pct_change')}%)",
        f"  • DXY {dxy.get('last')} ({dxy.get('pct_change', '0')}%)",
        f"  • 10-yr yield {tnx.get('last')} ({tnx.get('pct_change', '0')}%)",
        "",
        "Top Headlines:",
    ] + [f"  • {h}" for h in headlines] + [
        "",
        "Give a concise pre-market bias (bullish / bearish / neutral) for ES and NQ and key price levels to watch.",
    ]
    context = "\n".join(context_lines)

    # ----- ask GPT -----
    bias_text = summarize_with_openai(context)

    # ----- display -----
    st.subheader("📈 Bias Summary")
    st.write(bias_text)

    st.divider()
    with st.expander("🔍 Raw context sent to GPT"):
        st.code(context)

else:
    st.info("Enter your API keys (or rely on secrets.toml) and click **Run Generator**.")

# ---------------------------------------------------------------------------
# footer
# ---------------------------------------------------------------------------
st.markdown("---\n*Built with ❤️ and 🧠 by Lexi AI — July 2025*")
