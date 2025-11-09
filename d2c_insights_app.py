# streamlit_d2c_chat_app.py
# D2C Analytics Chat — GPT-5 (Trends, Comparisons & Charts)
# Upload your marketplace data, ask questions in natural language, and GPT-5 will create a step-by-step plan and analysis.

import streamlit as st
import pandas as pd
import openai
import json
import re

st.set_page_config(page_title="D2C Analytics Chat (GPT-5)", layout="wide")

# ---------------------- SETUP ----------------------

openai.api_key = st.secrets.get("OPENAI_API_KEY")

NUMERIC_COLS = [
    'Sale (Qty.)','Sale (Amount)','Return (Qty)','Return (Amount)',
    'GMV','Less Discount','Gross Revenue','Less Returns',
    'Gross Revenue (Inc. GST) Post Returns','Less GST','Net Revenue',
    'COGS Sales','COGS Returns','COGS Free Replacement',
    'Gross COGS','GM'
]

# ---------------------- DATA CLEANING ----------------------

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

# ---------------------- GPT-5 NATURAL LANGUAGE PARSER ----------------------

def llm_parse_question(question: str, df: pd.DataFrame) -> dict:
    available_markets = sorted(df['Marketplace'].dropna().unique().tolist()) if 'Marketplace' in df.columns else []
    available_months = sorted(df['Month'].dropna().unique().tolist()) if 'Month' in df.columns else []

    system_prompt = (
        "You are GPT-5, a world-class data analysis planner for a D2C analytics system.\n"
        "Given a user's natural language question and the dataset columns (month, marketplace, product name, sales, returns, revenue, COGS, GM),\n"
        "produce a structured JSON plan describing exactly how to answer the question using pandas.\n"
        "JSON must contain: { 'months': [list], 'marketplaces': [list], 'metric': 'string', 'top_n': int, 'group_by': [list], "
        "'operations': [list], 'visualization': 'string', 'steps': [list] }.\n"
        "If the question asks for trends, comparisons, or ratios, include logic such as grouping by month, computing % change, or calculating ratios.\n"
        "Valid visualizations: 'line_chart', 'bar_chart', 'pie_chart', 'table', or 'summary'.\n"
        "Default to showing all marketplaces and months if not specified."
    )

    user_prompt = f"Question: {question}\nAvailable Marketplaces: {available_markets}\nAvailable Months: {available_months}"

    response = openai.ChatCompletion.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    plan_text = response.choices[0].message['content']
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError:
        plan = {"error": "Failed to parse GPT-5 response", "raw": plan_text}
    return plan

# ---------------------- EXECUTION LOGIC ----------------------

def execute_plan(plan: dict, df: pd.DataFrame) -> dict:
    months = plan.get('months', [])
    markets = plan.get('marketplaces', [])
    metric = plan.get('metric', 'Net Revenue')
    top_n = plan.get('top_n', 5)
    group_by = plan.get('group_by', [])
    visualization = plan.get('visualization', 'table')

    # Filter by month and marketplace
    mask_month = df['Month'].str.contains('|'.join([re.escape(m) for m in months]), case=False, na=False) if months else pd.Series([True]*len(df))
    mask_market = df['Marketplace'].isin(markets) if markets else pd.Series([True]*len(df))
    df_filtered = df[mask_month & mask_market].copy()

    metric_list = ['Sale (Qty.)','Sale (Amount)','Return (Qty)','Return (Amount)','Net Revenue','Gross COGS','GM']
    use_metrics = [c for c in metric_list if c in df_filtered.columns]

    if not group_by:
        group_by = ['Product Name']

    grouped = df_filtered.groupby(group_by)[use_metrics].sum().reset_index()

    if metric in grouped.columns:
        grouped = grouped.sort_values(metric, ascending=False)

    top_products = grouped.head(top_n)

    # Trend or comparison logic
    if 'Month' in group_by:
        grouped['Month'] = pd.Categorical(grouped['Month'], ordered=True, categories=sorted(df['Month'].unique().tolist()))
        grouped = grouped.sort_values(['Product Name', 'Month'])
        grouped['MoM Change'] = grouped.groupby('Product Name')[metric].pct_change() * 100

    # Derived metrics
    if 'Return (Qty)' in grouped.columns and 'Sale (Qty.)' in grouped.columns:
        grouped['Return Rate %'] = (grouped['Return (Qty)'] / grouped['Sale (Qty.)'] * 100).round(2)

    # Gross margin percentage
    if 'Net Revenue' in grouped.columns and 'Gross COGS' in grouped.columns:
        grouped['GM%'] = (grouped['Net Revenue'] - grouped['Gross COGS']) / grouped['Net Revenue'] * 100

    return {
        'filtered_rows': len(df_filtered),
        'grouped': grouped,
        'top_products': top_products,
        'visualization': visualization,
        'metric': metric,
        'group_by': group_by
    }

# ---------------------- STREAMLIT UI ----------------------

st.title("D2C Analytics Chat — GPT-5 (Trends, Comparisons & Charts)")

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

# ✅ FIX for ambiguous DataFrame check
if raw_df is None or raw_df.empty:
    st.info("Please upload a valid dataset to begin.")
    st.stop()

df = clean_dataframe(raw_df)

st.subheader("Preview Cleaned Data")
st.dataframe(df.head(100))

st.subheader("Ask a question (interpreted by GPT-5)")
question = st.text_input("Example: 'Show the trend of net revenue from June to October across all marketplaces.'")

if st.button("Run Query") and question.strip():
    with st.spinner("GPT-5 is parsing your question..."):
        plan = llm_parse_question(question, df)
    st.markdown("### Step Plan (GPT-5 Output)")
    st.json(plan)

    if 'error' not in plan:
        with st.spinner("Executing analysis plan..."):
            result = execute_plan(plan, df)

        st.markdown("### Results")
        st.write(f"Filtered Rows: {result['filtered_rows']}")

        vis = result['visualization']
        grouped = result['grouped']
        metric = result['metric']

        if vis == 'line_chart' and 'Month' in grouped.columns:
            st.line_chart(grouped.set_index('Month')[metric])
        elif vis == 'bar_chart':
            st.bar_chart(grouped.set_index(result['group_by'][0])[metric])
        else:
            st.dataframe(grouped)

        st.download_button(
            label="Download Results CSV",
            data=grouped.to_csv(index=False).encode('utf-8'),
            file_name="analysis_results.csv"
        )

st.markdown("---")
st.caption("This version of the app uses GPT-5 to interpret natural language questions including trends, comparisons, ratios, and visual insights.")
