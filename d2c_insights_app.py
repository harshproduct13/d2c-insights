"""
D2C Insights — Streamlit App
--------------------------------
Upload your marketplace data (Amazon, Flipkart, Meesho, etc.)
and ask natural-language questions like:
- "Top 5 selling products by quantity on Meesho"
- "What was my top selling product in June?"
- "Which products have rising return % month-on-month?"

Run locally:
    pip install -r requirements.txt
    streamlit run d2c_insights_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import textwrap

# Optional plotting
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------
st.set_page_config(page_title="D2C Insights", layout="wide")
st.title("📊 D2C Insights — Upload + Ask")

st.markdown("""
Upload your marketplace sheet (Excel/CSV) and ask plain-English questions.
This prototype uses rule-based and optional LLM logic to translate questions into Pandas code.

---
""")


# ---------------- HELPERS ----------------
def read_uploaded_file(uploaded):
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            df = pd.read_excel(xls, xls.sheet_names[0])
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def normalize_colname(c: str):
    return (
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
    )


def summarize_columns(df):
    info = []
    for c in df.columns:
        info.append({
            "Column": c,
            "Type": str(df[c].dtype),
            "Non-null": df[c].notnull().sum(),
            "Sample": ", ".join(df[c].dropna().astype(str).head(3).tolist()),
        })
    st.table(pd.DataFrame(info))


# ---------------- SAFETY CHECK ----------------
ALLOWED_AST_NODES = (
    ast.Module, ast.Expr, ast.Load, ast.Store, ast.Assign,
    ast.Name, ast.Call, ast.Attribute, ast.Subscript,
    ast.Index, ast.Slice, ast.Tuple, ast.List, ast.Dict,
    ast.UnaryOp, ast.BinOp, ast.Compare, ast.IfExp,
    ast.Num, ast.Constant, ast.Str, ast.Return,
    ast.For, ast.While, ast.Break, ast.Continue,
    ast.keyword  # ✅ now allowed
)
ALLOWED_NAMES = {"df", "pd", "np"}


def is_code_safe(code: str):
    if "import " in code:
        return False, "Imports are not allowed."
    forbidden = ["__", "open(", "exec(", "eval(", "subprocess", "os.", "sys.", "requests"]
    for f in forbidden:
        if f in code:
            return False, f"Forbidden token detected: {f}"

    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"Failed to parse code: {e}"

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            return False, f"Disallowed AST node: {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and not node.id.isidentifier():
                return False, f"Disallowed variable name: {node.id}"
    return True, "Safe"


def execute_code(code: str, df: pd.DataFrame):
    ok, msg = is_code_safe(code)
    if not ok:
        return None, f"❌ Safety check failed: {msg}"

    safe_globals = {"pd": pd, "np": np}
    safe_locals = {"df": df.copy()}

    try:
        compiled = compile(code, "<string>", "exec")
        exec(compiled, safe_globals, safe_locals)
        for var in ["result", "out", "pivot", "grp", "tmp"]:
            if var in safe_locals:
                return safe_locals[var], "✅ Executed successfully"
        try:
            val = eval(code, safe_globals, {"df": df.copy(), "pd": pd, "np": np})
            return val, "✅ Executed successfully"
        except Exception:
            return None, "⚠️ No result variable found"
    except Exception as e:
        return None, f"Execution error: {e}"


# ---------------- QUERY → CODE GENERATOR ----------------
def generate_code(question: str, df: pd.DataFrame):
    q = question.lower()
    prod_col = next((c for c in df.columns if any(k in c.lower() for k in ["short", "name", "sku", "product"])), df.columns[0])
    qty_col = next((c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()), None)
    ret_qty_col = next((c for c in df.columns if "return" in c.lower() and "qty" in c.lower()), None)

    # --- Top selling product in month ---
    for m in ["june", "july", "august", "september", "october"]:
        if "top" in q and "sell" in q and m in q:
            return textwrap.dedent(f"""
                result = df[df['Month'].astype(str).str.lower().str.contains('{m}')].groupby('{prod_col}')[['{qty_col}']].sum().sort_values(by='{qty_col}', ascending=False).head(10)
            """)

    # --- Top N on marketplace ---
    if "meesho" in q and "top" in q and "qty" in q:
        return textwrap.dedent(f"""
            result = df[df['Marketplace'].astype(str).str.lower() == 'meesho'].groupby('{prod_col}')[['{qty_col}']].sum().sort_values(by='{qty_col}', ascending=False).head(5)
        """)

    # --- Returns rising ---
    if "return" in q and any(x in q for x in ["rising", "increasing", "rise"]):
        return textwrap.dedent(f"""
            tmp = df.copy()
            tmp['month_norm'] = tmp['Month'].astype(str)
            grp = tmp.groupby(['{prod_col}','month_norm'])[['{qty_col}','{ret_qty_col}']].sum().reset_index()
            grp['return_pct'] = grp['{ret_qty_col}'] / grp['{qty_col}'].replace(0, np.nan)
            pivot = grp.pivot(index='{prod_col}', columns='month_norm', values='return_pct').fillna(0)
            pivot['delta'] = pivot.iloc[:, -1] - pivot.iloc[:, 0]
            result = pivot.sort_values('delta', ascending=False).head(20)
        """)

    return "# Could not generate code for that question — try rephrasing."


# ---------------- STREAMLIT UI ----------------
uploaded = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "xls", "csv"])

if uploaded:
    df = read_uploaded_file(uploaded)
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))
        st.write(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

        st.subheader("Column Summary")
        summarize_columns(df)

        st.markdown("---")
        st.subheader("Ask a Question About Your Data")

        question = st.text_input("Type your question", value="What was my top selling Product in June?")
        if st.button("Generate Answer"):
            with st.spinner("Generating code..."):
                code = generate_code(question, df)
                st.code(code, language="python")

                result, msg = execute_code(code, df)
                if result is None:
                    st.error(msg)
                else:
                    st.success(msg)
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                        st.download_button("Download as CSV", result.to_csv().encode("utf-8"), "result.csv")

                        # If small numeric result, plot it
                        try:
                            if isinstance(result, pd.Series):
                                st.bar_chart(result)
                            elif result.shape[0] <= 30 and result.select_dtypes(include=[np.number]).shape[1] == 1:
                                st.bar_chart(result)
                        except Exception:
                            pass
                    else:
                        st.write(result)
else:
    st.info("Upload your file to start. Expected columns: Marketplace, Month, Shortened Name, SUM of Sale (Qty.), SUM of Return (Qty), SUM of GMV, SUM of Gross Revenue, etc.")
