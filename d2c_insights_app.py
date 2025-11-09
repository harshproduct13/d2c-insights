# app.py
# D2C Analytics Chat — GPT-5
# Natural language analytics with fuzzy matching and smart summaries

import streamlit as st
import pandas as pd
import json
import re
from difflib import get_close_matches
from openai import OpenAI

# -------------------- SETUP --------------------
st.set_page_config(page_title="D2C Analytics Chat (GPT-5)", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

NUMERIC_COLS = [
    'Sale (Qty.)','Sale (Amount)','Return (Qty)','Return (Amount)',
    'GMV','Less Discount','Gross Revenue','Less Returns',
    'Gross Revenue (Inc. GST) Post Returns','Less GST','Net Revenue',
    'COGS Sales','COGS Returns','COGS Free Replacement','Gross COGS','GM'
]

# -------------------- DATA CLEANING --------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str).str.strip()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(',', '').str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    if 'Marketplace' not in df.columns:
        df['Marketplace'] = 'Unknown'
    if 'Product Name' not in df.columns:
        df['Product Name'] = 'Unknown'
    return df

# -------------------- FUZZY MATCHING --------------------
def fuzzy_fix_plan(plan: dict, df: pd.DataFrame) -> dict:
    valid_cols = [c.strip() for c in df.columns.tolist()]

    def best_match(value):
        if not isinstance(value, str):
            return value
        match = get_close_matches(value.strip(), valid_cols, n=1, cutoff=0.6)
        return match[0] if match else value

    fuzzy_log = {}

    # Fix metric
    metric = plan.get('metric', '')
    fixed_metric = best_match(metric)
    fuzzy_log['metric'] = {'input': metric, 'fixed': fixed_metric}
    plan['metric'] = fixed_metric

    # Fix group_by
    if 'group_by' in plan and isinstance(plan['group_by'], list):
        corrected = []
        for g in plan['group_by']:
            fixed = best_match(g)
            if fixed in valid_cols:
                corrected.append(fixed)
            fuzzy_log[g] = fixed
        plan['group_by'] = corrected

    if not plan.get('group_by'):
        plan['group_by'] = ['Product Name'] if 'Product Name' in valid_cols else [valid_cols[0]]

    if 'operations' in plan and isinstance(plan['operations'], list):
        plan['operations'] = [best_match(op) for op in plan['operations']]

    if plan['metric'] not in valid_cols:
        plan['metric'] = 'Net Revenue' if 'Net Revenue' in valid_cols else valid_cols[-1]

    plan['_fuzzy_log'] = fuzzy_log
    return plan

# -------------------- GPT-5 PLAN PARSER --------------------
def llm_parse_question(question: str, df: pd.DataFrame) -> dict:
    available_cols = list(df.columns)
    available_markets = sorted(df['Marketplace'].dropna().unique().tolist()) if 'Marketplace' in df.columns else []
    available_months = sorted(df['Month'].dropna().unique().tolist()) if 'Month' in df.columns else []

    system_prompt = (
        "You are GPT-5, a world-class data analysis planner for a D2C analytics system.\n"
        "Given a user's natural-language question and dataset columns, "
        "produce a structured JSON plan describing how to answer the question using pandas.\n"
        "JSON must contain: { 'months':[list], 'marketplaces':[list], 'metric':'string', "
        "'top_n':int, 'group_by':[list], 'operations':[list], 'visualization':'string', 'steps':[list] }.\n"
        "Use column names similar to those provided; small naming differences are fine."
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Available columns: {available_cols}\n"
        f"Available marketplaces: {available_markets}\n"
        f"Available months: {available_months}"
    )

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    plan_text = response.choices[0].message.content
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse GPT-5 response", "raw": plan_text}

    return fuzzy_fix_plan(plan, df)

# -------------------- EXECUTION --------------------
def execute_plan(plan: dict, df: pd.DataFrame) -> dict:
    months = plan.get('months', [])
    markets = plan.get('marketplaces', [])
    metric = plan.get('metric', 'Net Revenue')
    top_n = plan.get('top_n', 5)
    group_by = plan.get('group_by', [])
    visualization = plan.get('visualization', 'table')

    mask_month = (
        df['Month'].str.contains('|'.join([re.escape(m) for m in months]), case=False, na=False)
        if months else pd.Series([True] * len(df))
    )
    mask_market = (
        df['Marketplace'].isin(markets)
        if markets else pd.Series([True] * len(df))
    )
    df_filtered = df[mask_month & mask_market].copy()

    metric_list = [
        'Sale (Qty.)','Sale (Amount)','Return (Qty)','Return (Amount)',
        'Net Revenue','Gross COGS','GM'
    ]
    use_metrics = [c for c in metric_list if c in df_filtered.columns]

    valid_group_by = [g for g in group_by if g in df_filtered.columns]
    if not valid_group_by:
        valid_group_by = ['Product Name']

    grouped = df_filtered.groupby(valid_group_by)[use_metrics].sum().reset_index()

    if metric in grouped.columns:
        grouped = grouped.sort_values(metric, ascending=False)

    top_products = grouped.head(top_n)

    # Derived metrics
    if 'Return (Qty)' in grouped.columns and 'Sale (Qty.)' in grouped.columns:
        grouped['Return Rate (%)'] = (grouped['Return (Qty)'] / grouped['Sale (Qty.)'] * 100).round(2)
    if 'Net Revenue' in grouped.columns and 'Gross COGS' in grouped.columns:
        grouped['GM%'] = ((grouped['Net Revenue'] - grouped['Gross COGS']) / grouped['Net Revenue'] * 100).round(2)

    return {
        'filtered_rows': len(df_filtered),
        'grouped': grouped,
        'top_products': top_products,
        'visualization': visualization,
        'metric': metric,
        'group_by': valid_group_by,
        'fuzzy_log': plan.get('_fuzzy_log', {})
    }

# -------------------- NATURAL-LANGUAGE ANSWER --------------------
def generate_natural_language_answer(question, result_df, metric_col="Net Revenue", top_n=5):
    if result_df.empty:
        return "No results found for your query."

    display_df = result_df.head(top_n)
    summary_records = []
    for _, row in display_df.iterrows():
        name = row.get('Product Name', 'Unknown')
        val = row.get(metric_col, None)
        if pd.notna(val):
            summary_records.append(f"{name} — ₹{val:,.2f}")

    data_json = json.dumps(summary_records, indent=2)
    prompt = f"""
    The user asked: "{question}"
    Here are the top results:
    {data_json}

    Write a short, clear, natural-language summary mentioning the products and their {metric_col} values in ₹.
    Begin with a sentence like 'Here are the top 5 selling products in July in terms of net revenue:'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a business analyst summarizing data clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

# -------------------- STREAMLIT UI --------------------
st.title("D2C Analytics Chat — GPT-5 (Natural Language Answers + Fuzzy Matching)")

with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your D2C data (CSV/XLSX)", type=['csv','xlsx'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        st.success(f"Loaded {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    else:
        st.info("Please upload your dataset to begin.")
        raw_df = None

if raw_df is None or raw_df.empty:
    st.stop()

df = clean_dataframe(raw_df)

st.subheader("Preview Cleaned Data")
st.dataframe(df.head(100))

st.subheader("Ask a Question (Natural Language)")
question = st.text_input("e.g., 'Show top 5 products by net revenue in July across all marketplaces.'")

if st.button("Run Query") and question.strip():
    with st.spinner("GPT-5 is understanding your question..."):
        plan = llm_parse_question(question, df)
    st.markdown("### Plan Generated by GPT-5")
    st.json(plan)

    if 'error' not in plan:
        with st.spinner("Running analysis..."):
            result = execute_plan(plan, df)

        st.markdown("### Results")
        st.write(f"Filtered Rows: {result['filtered_rows']}")

        # 🗣️ Generate and display natural-language answer
        answer_text = generate_natural_language_answer(question, result['top_products'], metric_col=result['metric'])
        st.markdown(answer_text)

        # Optional detailed data view
        with st.expander("View Detailed Data Table"):
            st.dataframe(result['top_products'])

        with st.expander("Fuzzy Matching Corrections"):
            st.json(result['fuzzy_log'])

        st.download_button(
            label="Download Results CSV",
            data=result['top_products'].to_csv(index=False).encode('utf-8'),
            file_name="analysis_results.csv"
        )

st.markdown("---")
st.caption("Powered by GPT-5 — natural-language analytics for D2C businesses.")
