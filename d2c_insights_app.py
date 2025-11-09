# app.py
# D2C Analytics Chat — GPT-5 (Natural Language, Fuzzy Filters, Diagnostics, and Narrative Results)

import streamlit as st
import pandas as pd
import json, re, numpy as np
from difflib import get_close_matches
from openai import OpenAI

st.set_page_config(page_title="D2C Analytics Chat — GPT-5", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

NUMERIC_COLS = [
    "Sale (Qty.)","Sale (Amount)","Return (Qty)","Return (Amount)",
    "GMV","Less Discount","Gross Revenue","Less Returns",
    "Gross Revenue (Inc. GST) Post Returns","Less GST","Net Revenue",
    "COGS Sales","COGS Returns","COGS Free Replacement","Gross COGS","GM"
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_text_col(s: pd.Series):
    return s.astype(str).str.lower().str.strip().str.replace(r"[^a-z0-9\s\-]", "", regex=True)

def normalize_month_string(month_str):
    """Normalize month names like '9-September', '09-Sep', 'Sept' -> 'september'."""
    month_str = str(month_str).lower()
    month_str = re.sub(r"[^a-z]", "", month_str)
    mapping = {
        "jan": "january","feb": "february","mar": "march","apr": "april",
        "may": "may","jun": "june","jul": "july","aug": "august",
        "sep": "september","sept": "september","oct": "october",
        "nov": "november","dec": "december"
    }
    for k, v in mapping.items():
        if month_str.startswith(k):
            return v
    return month_str

def fuzzy_lookup_candidates(candidate_list, valid_values, cutoff=0.5):
    mapping = {}
    valid_lower = [str(v).lower() for v in valid_values]
    for c in candidate_list:
        cstr = str(c).lower()
        matches = [v for v in valid_values if cstr in str(v).lower()]
        if matches:
            mapping[c] = matches[0]; continue
        close = get_close_matches(cstr, valid_lower, n=1, cutoff=cutoff)
        if close:
            idx = valid_lower.index(close[0])
            mapping[c] = valid_values[idx]
        else:
            mapping[c] = None
    return mapping

def clean_dataframe(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
    for col in ["Month","Product Name","Marketplace"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

# ---------------------------------------------------------------------
# GPT-5 Planner
# ---------------------------------------------------------------------
def llm_parse_question(question, df):
    system_prompt = (
        "You are GPT-5, a D2C analytics planner. "
        "Given a user's natural-language question and dataset columns, "
        "produce a JSON plan with keys: months, marketplaces, metric, top_n, "
        "group_by, operations, steps. Use similar but not exact column names."
    )
    user_prompt = f"Question: {question}\nAvailable columns: {list(df.columns)}"
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]
    )
    text = response.choices[0].message.content
    try:
        plan = json.loads(text)
    except:
        plan = {"error":"Failed to parse GPT-5 output","raw":text}
    plan["question_text"] = question
    return plan

# ---------------------------------------------------------------------
# Robust Execution with Fuzzy Filters
# ---------------------------------------------------------------------
def execute_plan_robust(plan, df):
    df2 = df.copy()
    df2["_month_norm"] = df2["Month"].apply(normalize_month_string) if "Month" in df2.columns else ""
    df2["_product_norm"] = normalize_text_col(df2["Product Name"]) if "Product Name" in df2.columns else ""
    df2["_marketplace_norm"] = normalize_text_col(df2["Marketplace"]) if "Marketplace" in df2.columns else ""

    months_plan = [m.strip() for m in plan.get("months",[]) if m]
    markets_plan = [m.strip() for m in plan.get("marketplaces",[]) if m]

    # ---------------- smarter product extraction ----------------
    product_candidates = []
    for op in plan.get("operations",[]):
        if "product" in str(op).lower():
            m = re.search(r"['\"]([^'\"]+)['\"]", str(op))
            if m: product_candidates.append(m.group(1))
    if not product_candidates:
        q = plan.get("question_text","")
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q)
        if caps: product_candidates = [caps[0]]
    if not product_candidates:
        product_candidates = []

    # ---------------- fuzzy maps ----------------
    month_map = fuzzy_lookup_candidates(months_plan, df2["Month"].unique().tolist(), 0.4) if months_plan else {}
    market_map = fuzzy_lookup_candidates(markets_plan, df2["Marketplace"].unique().tolist(), 0.5) if markets_plan else {}
    prod_map = fuzzy_lookup_candidates(product_candidates, df2["Product Name"].unique().tolist(), 0.5) if product_candidates else {}

    # ---------------- masks ----------------
    def make_mask(col_norm, patterns, mapping):
        if not patterns: return pd.Series(True, index=df2.index)
        mask = pd.Series(False, index=df2.index)
        for p in patterns:
            mapped = mapping.get(p)
            token = re.sub(r"[^a-z]","",str(mapped or p).lower())
            mask |= df2[col_norm].str.contains(token, na=False)
        return mask.fillna(False)

    m_mask = make_mask("_month_norm", months_plan, month_map)
    mk_mask = make_mask("_marketplace_norm", markets_plan, market_map)
    p_mask = make_mask("_product_norm", product_candidates, prod_map)

    combined = m_mask & mk_mask & p_mask
    df_fin = df2[combined].copy()

    debug = {
        "rows_total": len(df2),
        "months_plan": months_plan,
        "month_map": month_map,
        "markets_plan": markets_plan,
        "market_map": market_map,
        "product_candidates": product_candidates,
        "prod_map": prod_map,
        "matched_after_month": int(m_mask.sum()),
        "matched_after_market": int(mk_mask.sum()),
        "matched_after_product": int(p_mask.sum()),
        "matched_all": int(combined.sum()),
    }

    # ---------------- fallback: relax if empty ----------------
    if df_fin.empty:
        relaxed = df2[mk_mask & p_mask]
        if len(relaxed) > 0:
            st.warning("No exact match for month — showing results ignoring month filter.")
            df_fin = relaxed.copy()
        else:
            return {"empty":True,"debug":debug}

    if "Net Revenue" not in df_fin.columns:
        df_fin["Net Revenue"] = df_fin.get("Sale (Amount)",0) - df_fin.get("Return (Amount)",0)

    grouped = df_fin.groupby("Marketplace",as_index=False)["Net Revenue"].sum().sort_values("Net Revenue",ascending=False)

    summary = {r["Marketplace"]: float(r["Net Revenue"]) for _,r in grouped.iterrows()}
    diff_abs = diff_pct = None
    if len(grouped)>=2:
        a,b = grouped.iloc[0]["Net Revenue"], grouped.iloc[1]["Net Revenue"]
        diff_abs = a-b; diff_pct = diff_abs/b*100 if b!=0 else None

    return {"empty":False,"grouped":grouped,"summary":summary,
            "diff_abs":diff_abs,"diff_pct":diff_pct,"debug":debug}

# ---------------------------------------------------------------------
# Natural-Language Answer
# ---------------------------------------------------------------------
def generate_summary(question, result):
    if result.get("empty"): 
        return "No rows matched the filters. Please check product or month naming."
    summary = result["summary"]
    if not summary: return "No data found."
    items = "\n".join([f"- {k} — ₹{v:,.2f}" for k,v in summary.items()])
    answer = f"Here are the results for your question:\n\n{items}"
    if result["diff_abs"] is not None:
        if result["diff_abs"]>0:
            a,b = list(summary.keys())[:2]
            answer += f"\n\n{a} outperformed {b} by ₹{result['diff_abs']:,.2f} ({result['diff_pct']:.2f}%)."
        else:
            a,b = list(summary.keys())[:2]
            answer += f"\n\n{b} outperformed {a} by ₹{abs(result['diff_abs']):,.2f} ({abs(result['diff_pct']):.2f}%)."
    return answer

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("💬 D2C Analytics Chat — GPT-5 (Smart, Fuzzy & Conversational)")

with st.sidebar:
    st.header("Upload Dataset")
    file = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"])
    if not file:
        st.info("Please upload your dataset to begin.")
        st.stop()
    df_raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

df = clean_dataframe(df_raw)
st.subheader("📄 Data Preview")
st.dataframe(df.head())

question = st.text_input("Ask a question (e.g., 'What was the revenue of Hyaluronic Acid Serum on Meesho vs Flipkart in September?')")
if st.button("Run Query") and question.strip():
    with st.spinner("Analyzing question using GPT-5..."):
        plan = llm_parse_question(question, df)
    st.markdown("### 🧠 Plan Generated by GPT-5")
    st.json(plan)

    if "error" in plan:
        st.error("Failed to parse GPT-5 output.")
        st.stop()

    with st.spinner("Executing plan and analyzing data..."):
        result = execute_plan_robust(plan, df)

    st.markdown("### 📊 Results")
    answer = generate_summary(question, result)
    st.markdown(answer)

    if not result.get("empty"):
        with st.expander("View Detailed Data Table"):
            st.dataframe(result["grouped"])
    with st.expander("🧩 Diagnostics (for debugging)"):
        st.json(result["debug"])
