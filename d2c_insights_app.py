"""
D2C Insights — Streamlit App

Save as: d2c_insights_app.py
Run: pip install -r requirements.txt
     export OPENAI_API_KEY=sk-...
     streamlit run d2c_insights_app.py

Requirements (suggested):
streamlit
pandas
numpy
openai
python-dotenv
duckdb (optional)

Description:
- Upload marketplace Excel/CSV
- Ask natural language questions
- LLM (OpenAI) converts question to pandas code (or fallback heuristics)
- Code executed in a restricted sandbox and results shown as table or chart

WARNING: This is a prototype. Review generated code before using in production.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import ast
import textwrap
from typing import Tuple

# Optional import for charts
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
APP_TITLE = "D2C Insights — Upload + Ask (Prototype)"
MAX_PREVIEW_ROWS = 50
ALLOWED_AST_NODES = (
    ast.Module, ast.Expr, ast.Load, ast.Store, ast.Assign,
    ast.Name, ast.Call, ast.Attribute, ast.Subscript,
    ast.Index, ast.Slice, ast.Tuple, ast.List, ast.Dict,
    ast.UnaryOp, ast.BinOp, ast.Compare, ast.IfExp,
    ast.Num, ast.Constant, ast.Str, ast.Return,
    ast.For, ast.While, ast.Break, ast.Continue,
)
ALLOWED_NAMES = {"df", "pd", "np"}

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown("""
Upload your consolidated marketplace sheet (Excel or CSV). Then ask plain-English questions like:
- "Top 5 selling products by quantity on Meesho"
- "Which products have rising return% month-on-month?"
- "What was my top selling product in June?"

This is a developer prototype. The app will attempt to use an LLM (OpenAI) to convert questions to pandas code, but it includes safety checks. Use responsibly.
""")

# -----------------------------
# Helpers
# -----------------------------

def read_uploaded_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith('.csv'):
            return pd.read_csv(uploaded)
        else:
            # read first sheet
            xls = pd.ExcelFile(uploaded)
            return pd.read_excel(xls, xls.sheet_names[0])
    except Exception as e:
        st.error(f"Failed reading file: {e}")
        return None


def normalize_colname(c: str) -> str:
    return c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '').replace('/', '_')


def build_column_map(df: pd.DataFrame) -> Tuple[dict, dict]:
    orig_cols = list(df.columns)
    norm_map = {normalize_colname(c): c for c in orig_cols}
    return {c: normalize_colname(c) for c in orig_cols}, norm_map


def display_preview(df: pd.DataFrame):
    st.subheader("Dataset preview")
    st.dataframe(df.head(MAX_PREVIEW_ROWS))
    st.write(f"Rows: {len(df):,}, Columns: {len(df.columns)}")


def summarize_columns(df: pd.DataFrame):
    st.subheader("Column summary")
    info = []
    for c in df.columns:
        info.append({
            'column': c,
            'dtype': str(df[c].dtype),
            'non_null': int(df[c].notnull().sum()),
            'sample': ', '.join(map(str, df[c].dropna().astype(str).head(3).tolist()))
        })
    st.table(pd.DataFrame(info))


# Very small AST-based sanitizer
def is_code_safe(code: str) -> Tuple[bool, str]:
    if 'import ' in code:
        return False, "Imports not allowed in generated code"
    forbidden_tokens = ['__', 'open(', 'exec(', 'eval(', 'subprocess', 'os.', 'sys.', 'socket', 'requests', 'shutil']
    for t in forbidden_tokens:
        if t in code:
            return False, f"Forbidden token detected: {t}"
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"Unable to parse code: {e}"
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            return False, f"Disallowed AST node: {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and not node.id.isidentifier():
                return False, f"Disallowed name used: {node.id}"
    return True, "Safe"


# Execute code in restricted context
def execute_code(code: str, df: pd.DataFrame):
    ok, msg = is_code_safe(code)
    if not ok:
        return None, f"Safety check failed: {msg}"
    safe_globals = {"pd": pd, "np": np}
    safe_locals = {"df": df.copy()}
    try:
        compiled = compile(code, '<string>', 'exec')
        exec(compiled, safe_globals, safe_locals)
        # Fetch a named common variable
        for name in ['result', 'out', 'pivot', 'grp', 'tmp']:
            if name in safe_locals:
                return safe_locals[name], 'ok'
        # fallback: try eval (single expression)
        try:
            val = eval(code, safe_globals, {'df': df.copy(), 'pd': pd, 'np': np})
            return val, 'ok'
        except Exception:
            return None, 'Executed but no output variable found.'
    except Exception as e:
        return None, f'Execution error: {e}'


# Simple rule-based code generator for common patterns + LLM prompt builder

def generate_code(question: str, df: pd.DataFrame, openai_api_key: str = None) -> str:
    q = question.strip().lower()
    # identify product-like column
    prod_candidates = [c for c in df.columns if any(k in c.lower() for k in ['short', 'product', 'sku', 'name'])]
    prod_col = prod_candidates[0] if prod_candidates else df.columns[0]
    qty_candidates = [c for c in df.columns if ('qty' in c.lower() or 'quantity' in c.lower() or ('sale' in c.lower() and 'qty' in c.lower()))]
    qty_col = qty_candidates[0] if qty_candidates else None
    ret_qty_candidates = [c for c in df.columns if ('return' in c.lower() and 'qty' in c.lower())]
    ret_qty_col = ret_qty_candidates[0] if ret_qty_candidates else None

    # RULE: Top selling product in month
    if 'top' in q and ('selling' in q or 'sold' in q) and any(m in q for m in ['june','july','august','september','october']):
        # extract month word
        for m in ['june','july','august','september','october']:
            if m in q:
                month = m
                break
        if qty_col:
            code = textwrap.dedent(f"""
                result = df[df['Month'].astype(str).str.lower().str.contains('{month}')].groupby('{prod_col}')[['{qty_col}']].sum().sort_values(by='{qty_col}', ascending=False).head(10)
            """)
            return code
    # RULE: Top N selling by quantity on marketplace
    if 'top' in q and ('selling' in q or 'sold' in q) and 'meesho' in q:
        n = 5
        if 'top 5' in q:
            n = 5
        if qty_col:
            code = textwrap.dedent(f"""
                result = df[df['Marketplace'].astype(str).str.lower()=='meesho'].groupby('{prod_col}')[['{qty_col}']].sum().sort_values(by='{qty_col}', ascending=False).head({n})
            """)
            return code
    # RULE: returns rising as percentage
    if 'returns' in q and ('rising' in q or 'increasing' in q or 'rise' in q):
        if qty_col and ret_qty_col:
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['month_norm'] = tmp['Month'].astype(str)
                grp = tmp.groupby(['{prod_col}','month_norm'])[['{qty_col}','{ret_qty_col}']].sum().reset_index()
                grp['return_pct'] = grp['{ret_qty_col}'] / grp['{qty_col}'].replace(0, np.nan)
                pivot = grp.pivot(index='{prod_col}', columns='month_norm', values='return_pct').fillna(0)
                pivot['delta'] = pivot.iloc[:, -1] - pivot.iloc[:, 0]
                result = pivot.sort_values('delta', ascending=False).head(20)
            """)
            return code

    # If OpenAI key provided, build a prompt and call LLM
    openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            cols_snippet = '\n'.join([f'- {c}' for c in df.columns])
            system = textwrap.dedent(f"""
                You are a helpful data analyst. You're given a pandas DataFrame variable named `df` with these columns:\n{cols_snippet}\n
                Only respond with Python pandas code that uses `df` and `pd`/`np` and returns the final result as a variable named `result`.
                Do NOT import modules or access files or network. Do not print. Use safe pandas operations.
            """)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}"}
            ]
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=messages,
                max_tokens=500,
                temperature=0
            )
            code = resp.choices[0].message.content
            return code
        except Exception as e:
            st.warning(f"LLM call failed: {e}")

    return "# Sorry — could not generate code for that question. Try a different phrasing or provide OpenAI API key for LLM assistance."


# -----------------------------
# Streamlit UI
# -----------------------------

uploaded = st.file_uploader("Upload your marketplace sheet (Excel/CSV)", type=['xlsx','xls','csv'])

if uploaded:
    df = read_uploaded_file(uploaded)
    if df is None:
        st.stop()

    # Normalize month column if possible (helpful)
    if 'Month' in df.columns:
        try:
            df['Month'] = df['Month'].astype(str)
        except Exception:
            pass

    display_preview(df)
    summarize_columns(df)
    orig_to_norm, norm_to_orig = build_column_map(df)

    st.subheader("Detected columns (normalized preview)")
    st.json(norm_to_orig)

    st.markdown('---')
    st.subheader('Ask natural-language question')
    question = st.text_input('Type your question', value='What was my top selling Product in June?')
    openai_key_input = st.text_input('OpenAI API key (optional — leave blank to use environment variable)', type='password')

    if st.button('Generate Answer'):
        with st.spinner('Generating...'):
            code = generate_code(question, df, openai_api_key=openai_key_input or None)
            st.subheader('Generated code')
            st.code(code, language='python')
            out, status = execute_code(code, df)
            if out is None:
                st.error(status)
            else:
                st.success('Execution result')
                # Display output
                if isinstance(out, pd.DataFrame) or isinstance(out, pd.Series):
                    st.dataframe(out)
                    csv = out.to_csv(index=True).encode('utf-8')
                    st.download_button('Download CSV', csv, file_name='query_result.csv')
                    # if likely numeric groupby, show a chart
                    try:
                        if isinstance(out, pd.Series):
                            st.line_chart(out)
                        else:
                            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
                            if len(numeric_cols) == 1 and out.shape[0] <= 50:
                                st.bar_chart(out[numeric_cols[0]])
                    except Exception:
                        pass
                else:
                    st.write(out)

else:
    st.info('Upload a file to begin. Expected columns (examples): Marketplace, Month, Shortened Name, SUM of Sale (Qty.), SUM of Sale (Amount), SUM of Return (Qty), SUM of Return (Amount), SUM of GMV, SUM of Less Discount, SUM of Gross Revenue, SUM of Less Returns, SUM of Gross Revenue (Inc. GST) Post Returns, SUM of Less GST, SUM of Net Revenue, SUM of COGS Sales, SUM of COGS Returns, SUM of COGS Free Replacement, SUM of Gross COGS, SUM of GM')

# Footer
st.markdown('---')
st.caption('Prototype — review generated code before running in production. Need custom features (persistent storage, authentication, production sandboxing)? Reply and I will add them.')
