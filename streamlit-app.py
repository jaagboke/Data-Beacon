import os
import tempfile

import streamlit as st
import pandas as pd

from file_automation import (
    count_rows,
    human_readable_size,
    infer_schema,
    format_schema,
    format_preview,
    scan_chunks,
    format_numeric_stats,
    format_top_values,
)

st.set_page_config(page_title="CSV Inspector", layout="wide")
st.title("📊 File Inspector")

# 1) File Uploader in the sidebar
st.sidebar.header("1. Upload a CSV file")
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop CSV here or click to browse",
    type=["csv"],
    accept_multiple_files=False,
)

if not uploaded_file:
    st.sidebar.info("Please upload a CSV file to begin.")
    st.stop()

# 2) Write uploaded file to a temp file, so the existing code (which expects a file path) works as-is
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

# 3) Show basic file info
file_size = os.path.getsize(tmp_path)
row_count = count_rows(tmp_path)
col1, col2 = st.columns(2)
with col1:
    st.metric(label="File Size", value=human_readable_size(file_size))
with col2:
    st.metric(label="Row Count", value=f"{row_count:,}")

# 4) Infer schema and show a small preview
st.sidebar.header("2. Preview & Schema")
with st.spinner("Reading first 5,000 rows to infer schema…"):
    schema, df_head = infer_schema(tmp_path, nrows=5000)

st.subheader("Data Preview (first 10 rows)")
df_preview = format_preview(df_head, max_rows=10)
if df_preview.empty:
    st.write("(no data to preview)")
else:
    st.dataframe(df_preview)

st.subheader("Inferred Schema (from first 5,000 rows)")
df_schema = pd.DataFrame({
    "Column": list(schema.keys()),
    "Dtype": [schema[c] for c in schema.keys()]
})
st.dataframe(df_schema)

# 5) Button to run the full scan
st.sidebar.header("3. Compute Full Statistics")
top_n = st.sidebar.number_input(
    "Top-N for categorical columns", min_value=1, max_value=100, value=5
)
run_scan = st.sidebar.button("Run Full Scan (may take a while)")

# Placeholders
if run_scan:
    with st.spinner("Scanning CSV in chunks…"):
        null_counts, numeric_summary, top_values = scan_chunks(
            tmp_path,
            schema,
            chunk_size=50_000,
            top_k=top_n
        )

    # 6) Display Null Counts
    st.subheader("Null Counts (per column)")
    df_nulls = pd.DataFrame({
        "Column": list(null_counts.keys()),
        "Null Count": list(null_counts.values())
    })
    df_nulls["Null Count"] = df_nulls["Null Count"].map(lambda x: f"{x:,}")
    st.dataframe(df_nulls)

    # 7) Display Numeric Summary (min, max, mean)
    st.subheader("Numeric Columns: Min, Max, Mean")
    df_num = pd.DataFrame([
        {"Column": col, "Min": stats["min"], "Max": stats["max"], "Mean": stats["mean"]}
        for col, stats in numeric_summary.items()
        if stats["min"] is not None
    ])
    st.dataframe(df_num.style.format({
        "Min": "{:.4g}",
        "Max": "{:.4g}",
        "Mean": "{:.4g}"
    }))

    # 8) Display Histograms
    st.subheader("Numeric Histograms")
    for col, stats in numeric_summary.items():
        hist = stats.get("histogram", [])
        if not hist:
            st.write(f"**{col}**: (no histogram data)")
            continue

        df_hist = pd.DataFrame([{
            "bin_mid": (b["bin_start"] + b["bin_end"]) / 2,
            "count": b["count"]
        } for b in hist])

        st.write(f"**{col}**")
        df_hist = df_hist.set_index("bin_mid")
        st.bar_chart(df_hist["count"])

    # 9) Display Categorical Top-N Values
    st.subheader("Dimension Columns: Distinct & Top-N Values")
    for col, values in top_values.items():
        distinct = len(values)  # true distinct would be len(Counter), but top_values is only the top N
        st.write(f"**{col}**")
        if not values:
            st.write("  (no non-null values found)")
            continue
        df_top = pd.DataFrame(values, columns=["Value", "Count"])
        df_top["Count"] = df_top["Count"].map(lambda x: f"{x:,}")
        st.table(df_top)

    # 10) Show larger sample (first 1,000 rows of df_head if available)
    st.subheader("Sample Rows (up to 1,000 from first chunk)")
    n_preview = min(1000, len(df_head))
    if n_preview > 0:
        st.dataframe(df_head.sample(n_preview))
    else:
        st.write("(no data to sample)")
else:
    st.info("Click “Run Full Scan” in the sidebar to compute statistics.")


# 11) Cleanup: remove the temp file when the app shuts down
def cleanup_tempfile(path):
    try:
        os.remove(path)
    except Exception:
        pass

