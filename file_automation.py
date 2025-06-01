#!/usr/bin/penv python3
import argparse
import os
import subprocess
import sys
from collections import Counter
from math import inf, isfinite
import pandas as pd
import numpy as np

"""
This function counts how many data rows are in the file.
If the file is ran on windows. File is open in binary mode, read 1MiB at a time/
"""
def count_rows(path):
    """
    :param path: Pass the file path
    :return: Return number of data rows (subtracting 1 for header)
    """
    try:
        result = subprocess.run(
            ["wc -l", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        total_lines = int(result.stdout.strip().split()[0])
        #Subtract 1 to exclude the header row
        return max(0, total_lines, -1)
    except Exception:
        #If 'wc' is not available, read in binary chunks
        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20),b""): #read om 1 MiB blocks
                count += chunk.count(b"\n")
            #subtract 1 for header
            return max(0, count - 1)

"""
This function takes the raw dictionaries from the file and format them to into neat, aligned text tables.
"""
def human_readable_size(num_bytes):
    for unit in ["B","KB","MB","GB","TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

"""
This function returns a multi lines string of an aligned table
"""
def format_schema(schema, null_counts):
    lines = []
    lines.append(f"{'Column':<30} {'Dtype':<10} {'Nulls':>10}")
    lines.append("-"*30 + " " + "-"*10 + " " + "-"*10)
    for col, dtype in schema.items():
        n_null = null_counts.get(col, 0)
        lines.append(f"{col:<30} {dtype:<10} {n_null:>10,}")
    return "\n".join(lines)

"""
This function formats the numeric stats and returns a multi line string.
The numbers are formatted (up to 4 significant figures).
"""
def format_numeric_stats(numeric_summary):
    lines = []
    if not numeric_summary:
        return "  (no numeric columns detected)\n"
    lines.append(f"{'Numeric Column':<30} {'Min':>15} {'Max':>15} {'Mean':>15}")
    lines.append("-" * 30 + " " + "-" * 15 + " " + "-" * 15 + " " + "-" * 15)
    for col, stats in numeric_summary.items():
        mn = stats["min"]
        mx = stats["max"]
        mean = stats["mean"]
        mn_s = f"{mn:.4g}" if mn is not None else "-"
        mx_s = f"{mx:.4g}" if mx is not None else "-"
        mean_s = f"{mean:.4g}" if mean is not None else "-"
        lines.append(f"{col:<30} {mn_s:>15} {mx_s:>15} {mean_s:>15}")
    return "\n".join(lines)

"""
This function formats the top values in a multi string like
"""
def format_top_values(top_values):
    lines = []
    if not top_values:
        return "  (no categorical columns detected)\n"
    for col, vals in top_values.items():
        lines.append(f"{col}:")
        if not vals:
            lines.append("    (no non-null values detected)")
        else:
            for val, cnt in vals:
                vstr = str(val)
                if len(vstr) > 30:
                    vstr = vstr[:27] + "..."
                lines.append(f"    {vstr:<30} {cnt:>10,}")
        lines.append("")  # blank line after each column
    return "\n".join(lines)

"""
This function formats the preview. It returns the first 'max_rows' as a string table.
The function calls the head and hides the index column
"""
def format_preview(df, max_rows=10):
    if df.empty:
        return "(no data to preview)\n"
    return df.head(max_rows).to_string(index=False) + "\n"


"""
This function reads the first rows via pandas to infer column names + dtypes
"""
def infer_schema(path, nrows=5000):
    """
        :param path: Pass the file path and row number
        :return: Return (schema_dict, preview_df)
    """
    df_head = pd.read_csv(path, nrows=nrows)
    dtypes = df_head.dtypes.to_dict()
    schema = {col: str(dtypes[col]) for col in df_head.columns}
    return schema, df_head

"""
This function is a scan of the entire csv in chunks to get the 
1) null count
2) numeric summary of the data
3) top values
"""
def scan_chunks(path, schema, chunk_size=50_000, top_k=5):
    """
    One-pass scan of the entire CSV in chunks to compute:
      1) null_counts:   { column: total_nulls }
      2) numeric_summary: { column: { min, max, mean, histogram } }
      3) top_values:    { column: [(value, count), ... up to top_k] }
    Returns:
      - null_counts (dict)
      - numeric_summary (dict)
      - top_values (dict)
    """
    columns = list(schema.keys())
    # Initialize null counters at zero
    null_counts = {col: 0 for col in columns}

    # Split columns by numeric vs. categorical
    numeric_cols = [col for col, dt in schema.items() if dt.startswith(("int", "float"))]
    cat_cols = [col for col in columns if col not in numeric_cols]

    # For numeric: keep sum, count, min, max, and a small reservoir of samples
    numeric_sample_limit = 50_000
    num_acc = {}
    for col in numeric_cols:
        num_acc[col] = {
            "sum": 0.0,
            "count": 0,
            "min": inf,
            "max": -inf,
            "samples": []   # reservoir to build histogram later
        }

    # For categorical: keep a Counter to track frequencies
    cat_counters = {col: Counter() for col in cat_cols}

    # Iterate in chunks
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        # 1) NULL COUNTS
        chunk_nulls = chunk.isna().sum()
        for col in columns:
            null_counts[col] += int(chunk_nulls.get(col, 0))

        # 2) NUMERIC STATS
        for col in numeric_cols:
            series = chunk[col].dropna()  # drop missing values
            if series.empty:
                continue
            arr = series.values.astype(float)  # convert to NumPy array
            s_sum = float(arr.sum())
            s_count = int(arr.size)
            s_min = float(np.min(arr))
            s_max = float(np.max(arr))

            acc = num_acc[col]
            acc["sum"] += s_sum
            acc["count"] += s_count
            if s_min < acc["min"]:
                acc["min"] = s_min
            if s_max > acc["max"]:
                acc["max"] = s_max

            # RESERVOIR SAMPLING FOR HISTOGRAM
            for v in arr:
                if len(acc["samples"]) < numeric_sample_limit:
                    acc["samples"].append(v)
                else:
                    idx = np.random.randint(0, acc["count"])
                    if idx < numeric_sample_limit:
                        acc["samples"][idx] = v

        # 3) CATEGORICAL COUNTS
        for col in cat_cols:
            series = chunk[col].dropna().astype(str)
            if series.empty:
                continue
            cat_counters[col].update(series.values)

    # AFTER READING ALL CHUNKS → BUILD FINAL SUMMARIES

    # A) Numeric summary
    numeric_summary = {}
    for col, acc in num_acc.items():
        if acc["count"] == 0:
            numeric_summary[col] = {
                "min": None,
                "max": None,
                "mean": None,
                "histogram": []
            }
        else:
            mean = acc["sum"] / acc["count"]
            samples = np.array(acc["samples"]) if acc["samples"] else np.array([])
            if samples.size > 0:
                # Build a 20-bin histogram from the reservoir of samples
                hist_counts, bin_edges = np.histogram(samples, bins=20)
                histogram = [
                    {
                        "bin_start": float(bin_edges[i]),
                        "bin_end":   float(bin_edges[i+1]),
                        "count":     int(hist_counts[i])
                    }
                    for i in range(len(hist_counts))
                ]
            else:
                histogram = []
            mn = acc["min"] if isfinite(acc["min"]) else None
            mx = acc["max"] if isfinite(acc["max"]) else None
            numeric_summary[col] = {
                "min": mn,
                "max": mx,
                "mean": mean,
                "histogram": histogram
            }

    # B) Top-k values for categorical columns
    top_values = {}
    for col, counter in cat_counters.items():
        top_values[col] = counter.most_common(top_k)

    return null_counts, numeric_summary, top_values



def main():
    #Initialise argument parser
    parser = argparse.ArgumentParser(
        description="Inspect a large CSV file (schema, row count, nulls, basic stats)."
    )
    parser.add_argument("file", help="Path to CSV file to inspect")
    parser.add_argument(
        "--preview", "-p", type=int, default=0,
        help="Show first N rows (in addition to stats)."
    )
    parser.add_argument(
        "--column", "-c", metavar="COLNAME", default=None,
        help="If specified, show only stats for that column."
    )
    parser.add_argument(
        "--no-scan", dest="no_scan", action="store_true",
        help="Skip the chunked scan (will only print schema & row count)."
    )
    args = parser.parse_args()

    csv_path = args.file
    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 1) File info
    file_size = os.path.getsize(csv_path)
    print(f"\nFile: {csv_path}")
    print(f"Size: {human_readable_size(file_size)}")

    # 2) Row count (fast)
    print("Counting rows...", end="", flush=True)
    row_count = count_rows(csv_path)
    print(f" {row_count:,} rows (excluding header)")

    # 3) Infer schema & (optional) preview
    print("Inferring schema (first 5,000 rows)...")
    schema, df_head = infer_schema(csv_path, nrows=5000)

    if args.preview > 0:
        print(f"\n=== Preview: first {args.preview} rows ===")
        print(format_preview(df_head, max_rows=args.preview))

    # 4) If --no-scan is set, skip computing nulls & stats
    if args.no_scan:
        print("\n=== Schema (no stats, --no-scan used) ===")
        for col, dt in schema.items():
            print(f"  • {col:<30} {dt}")
        sys.exit(0)

    # 5) One-pass scan
    print("Scanning in chunks to compute nulls & stats (may take a while)...")
    null_counts, numeric_summary, top_values = scan_chunks(csv_path, schema)

    # 6) If user asked for a single column
    if args.column:
        col = args.column
        if col not in schema:
            print(f"Error: column '{col}' not found in schema.", file=sys.stderr)
            sys.exit(1)
        dtype = schema[col]
        n_null = null_counts.get(col, 0)
        print(f"\n=== Stats for column '{col}' (dtype={dtype}) ===")
        print(f"  Null count: {n_null:,}")
        if dtype.startswith(("int", "float")):
            stats = numeric_summary[col]
            mn = stats["min"]
            mx = stats["max"]
            mean = stats["mean"]
            print(f"  Min:  {mn if mn is not None else '-'}")
            print(f"  Max:  {mx if mx is not None else '-'}")
            print(f"  Mean: {mean if mean is not None else '-'}")
            hist = stats["histogram"]
            if hist:
                print("\n  Histogram (up to first 5 bins):")
                for bin_info in hist[:5]:
                    bs = bin_info["bin_start"]
                    be = bin_info["bin_end"]
                    cnt = bin_info["count"]
                    print(f"    [{bs:.4g}, {be:.4g}) → {cnt:,}")
                if len(hist) > 5:
                    print("    ...")
        else:
            vals = top_values.get(col, [])
            print("\n  Top values:")
            if not vals:
                print("    (no non-null values found)")
            else:
                for val, cnt in vals:
                    vstr = val if len(str(val)) <= 40 else str(val)[:37] + "..."
                    print(f"    {vstr:<40} {cnt:,}")
        sys.exit(0)

    # 7) Otherwise, print full summary
    print("\n=== Schema & Null Counts ===")
    print(format_schema(schema, null_counts))

    print("\n=== Numeric Columns Summary ===")
    print(format_numeric_stats(numeric_summary))

    print("\n=== Top 5 Values for Categorical Columns ===")
    print(format_top_values(top_values))

    print("\nDone.\n")

if __name__ == "__main__":
    main()
