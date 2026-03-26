import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-Assisted Data Wrangler & Visualizer", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(name, content):
    buf = io.BytesIO(content)
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith(".xlsx"):
        return pd.read_excel(buf)
    if name.endswith(".json"):
        return pd.read_json(buf)
    raise ValueError("Upload CSV, XLSX, or JSON only.")


@st.cache_data(show_spinner=False)
def profile_data(df):
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values
    })
    num = df.describe(include=[np.number]).T if not df.select_dtypes(include=[np.number]).empty else pd.DataFrame()
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    cat = cat_df.describe().T if not cat_df.empty else pd.DataFrame()
    return info, num, cat


def init_state():
    defaults = {
        "df": None,
        "original_df": None,
        "history": [],
        "log": [],
        "violations": pd.DataFrame(),
        "file_name": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_state():
    for k in ["df", "original_df", "file_name"]:
        st.session_state[k] = None
    st.session_state["history"] = []
    st.session_state["log"] = []
    st.session_state["violations"] = pd.DataFrame()


def save_step(new_df, operation, params, cols):
    if st.session_state["df"] is not None:
        st.session_state["history"].append(st.session_state["df"].copy())
    st.session_state["df"] = new_df
    st.session_state["log"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "parameters": params,
        "affected_columns": cols
    })


def undo_step():
    if st.session_state["history"]:
        st.session_state["df"] = st.session_state["history"].pop()
        if st.session_state["log"]:
            st.session_state["log"].pop()


def has_data():
    if st.session_state["df"] is None:
        st.warning("Please upload a dataset first.")
        return False
    return True


def num_cols(df):
    return list(df.select_dtypes(include=["number"]).columns)


def cat_cols(df):
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def clean_numeric(series):
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(s, errors="coerce")


def outlier_mask(df, col, method):
    if method == "IQR":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        return (df[col] < low) | (df[col] > high)
    std = df[col].std(ddof=0)
    if std == 0:
        return pd.Series([False] * len(df), index=df.index)
    z = (df[col] - df[col].mean()) / std
    return (z < -3) | (z > 3)


def winsorize(series, low_q, high_q):
    low = series.quantile(low_q)
    high = series.quantile(high_q)
    return series.clip(lower=low, upper=high)


init_state()

page = st.sidebar.radio("Choose Page", [
    "Upload & Overview",
    "Cleaning Studio",
    "Visualization",
    "Export & Report"
])

st.sidebar.markdown("### Session Controls")
if st.sidebar.button("Reset session"):
    reset_state()
    st.sidebar.success("Session reset.")
if st.session_state["history"] and st.sidebar.button("Undo last step"):
    undo_step()
    st.sidebar.success("Last step undone.")


if page == "Upload & Overview":
    st.title("Upload & Overview")

    uploaded = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])
    if uploaded is not None:
        try:
            df = load_data(uploaded.name, uploaded.getvalue())
            st.session_state["df"] = df.copy()
            st.session_state["original_df"] = df.copy()
            st.session_state["history"] = []
            st.session_state["log"] = []
            st.session_state["violations"] = pd.DataFrame()
            st.session_state["file_name"] = uploaded.name
            st.success("File uploaded successfully.")
        except Exception as e:
            st.error(f"Could not load file: {e}")

    if has_data():
        df = st.session_state["df"]
        info, num_summary, cat_summary = profile_data(df)

        a, b, c = st.columns(3)
        a.metric("Rows", df.shape[0])
        b.metric("Columns", df.shape[1])
        c.metric("Duplicate rows", int(df.duplicated().sum()))

        st.subheader("Column types")
        st.dataframe(info[["column", "dtype"]], use_container_width=True)

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Missing values")
        st.dataframe(info[["column", "missing_count", "missing_pct"]], use_container_width=True)

        st.subheader("Summary statistics")
        if not num_summary.empty:
            st.write("Numeric summary")
            st.dataframe(num_summary, use_container_width=True)
        if not cat_summary.empty:
            st.write("Categorical summary")
            st.dataframe(cat_summary, use_container_width=True)


elif page == "Cleaning Studio":
    st.title("Cleaning Studio")

    if has_data():
        df = st.session_state["df"].copy()
        all_cols = list(df.columns)
        nums = num_cols(df)
        cats = cat_cols(df)

        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Missing Values")
        miss = pd.DataFrame({
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values
        })
        st.dataframe(miss, use_container_width=True)

        mcol = st.selectbox("Column for missing handling", all_cols)
        maction = st.selectbox("Action", [
            "Do nothing", "Drop rows", "Fill mean", "Fill median",
            "Fill mode", "Fill value", "Forward fill", "Backward fill"
        ])
        fill_val = st.text_input("Value to fill") if maction == "Fill value" else None

        if st.button("Apply Missing Handling"):
            new_df = df.copy()
            before_rows = len(df)
            before_missing = int(df[mcol].isna().sum())

            try:
                if maction == "Drop rows":
                    new_df = new_df.dropna(subset=[mcol])
                elif maction == "Fill mean":
                    if mcol not in nums:
                        st.error("Mean only works for numeric columns.")
                    else:
                        new_df[mcol] = new_df[mcol].fillna(new_df[mcol].mean())
                elif maction == "Fill median":
                    if mcol not in nums:
                        st.error("Median only works for numeric columns.")
                    else:
                        new_df[mcol] = new_df[mcol].fillna(new_df[mcol].median())
                elif maction == "Fill mode":
                    mode = new_df[mcol].mode(dropna=True)
                    if not mode.empty:
                        new_df[mcol] = new_df[mcol].fillna(mode.iloc[0])
                elif maction == "Fill value":
                    new_df[mcol] = new_df[mcol].fillna(fill_val)
                elif maction == "Forward fill":
                    new_df[mcol] = new_df[mcol].ffill()
                elif maction == "Backward fill":
                    new_df[mcol] = new_df[mcol].bfill()

                if maction != "Do nothing":
                    save_step(new_df, "missing_value_handling", {"column": mcol, "action": maction}, [mcol])
                    st.success(
                        f"Rows: {before_rows} → {len(new_df)} | Missing: {before_missing} → {int(new_df[mcol].isna().sum())}"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

        threshold = st.slider("Drop columns above missing threshold (%)", 0, 100, 50)
        if st.button("Drop Columns by Threshold"):
            try:
                pct = df.isna().mean() * 100
                drop = pct[pct > threshold].index.tolist()
                if not drop:
                    st.info("No columns exceed the threshold.")
                else:
                    new_df = df.drop(columns=drop)
                    save_step(new_df, "drop_columns_by_missing_threshold", {"threshold_pct": threshold}, drop)
                    st.success(f"Dropped columns: {drop}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("Duplicates")
        st.write("Full-row duplicates:", int(df.duplicated().sum()))
        dsubset = st.multiselect("Subset columns for duplicate check", all_cols)
        keep = st.selectbox("Keep", ["first", "last"])
        st.write("Subset duplicates:", int(df.duplicated(subset=dsubset).sum()) if dsubset else 0)

        if st.button("Show Duplicate Groups"):
            try:
                dup = df[df.duplicated(subset=dsubset if dsubset else None, keep=False)]
                if dsubset:
                    dup = dup.sort_values(by=dsubset)
                st.dataframe(dup, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Remove Duplicates"):
            try:
                new_df = df.drop_duplicates(subset=dsubset if dsubset else None, keep=keep)
                save_step(new_df, "remove_duplicates", {"subset": dsubset, "keep": keep}, dsubset)
                st.success(f"Removed {len(df) - len(new_df)} duplicate rows.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("Data Types & Parsing")
        tcol = st.selectbox("Column to convert", all_cols)
        ttype = st.selectbox("Convert to", ["numeric", "categorical", "datetime", "string"])
        dt_format = st.text_input("Datetime format (optional)", placeholder="%Y-%m-%d")
        if st.button("Convert Type"):
            try:
                new_df = df.copy()
                if ttype == "numeric":
                    new_df[tcol] = clean_numeric(new_df[tcol])
                elif ttype == "categorical":
                    new_df[tcol] = new_df[tcol].astype("category")
                elif ttype == "datetime":
                    new_df[tcol] = pd.to_datetime(new_df[tcol], format=dt_format or None, errors="coerce")
                else:
                    new_df[tcol] = new_df[tcol].astype(str)
                save_step(new_df, "convert_dtype", {"column": tcol, "target_type": ttype, "format": dt_format}, [tcol])
                st.success("Column converted.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("Categorical Cleaning")
        if cats:
            ccol = st.selectbox("Categorical column", cats)
            standard = st.selectbox("Standardize", ["None", "Lowercase", "Uppercase", "Title Case", "Trim Spaces"])
            if st.button("Apply Standardize"):
                try:
                    new_df = df.copy()
                    s = new_df[ccol].astype(str)
                    if standard == "Lowercase":
                        new_df[ccol] = s.str.lower()
                    elif standard == "Uppercase":
                        new_df[ccol] = s.str.upper()
                    elif standard == "Title Case":
                        new_df[ccol] = s.str.title()
                    elif standard == "Trim Spaces":
                        new_df[ccol] = s.str.strip()
                    save_step(new_df, "categorical_standardization", {"column": ccol, "method": standard}, [ccol])
                    st.success("Standardization applied.")
                except Exception as e:
                    st.error(f"Error: {e}")

            mapping_text = st.text_area("Mapping JSON", value='{"old_value": "new_value"}')
            set_other = st.checkbox("Set unmatched values to Other")
            if st.button("Apply Mapping"):
                try:
                    mapping = json.loads(mapping_text)
                    new_df = df.copy()
                    mapped = new_df[ccol].map(mapping)
                    new_df[ccol] = mapped.fillna("Other") if set_other else new_df[ccol].replace(mapping)
                    save_step(new_df, "categorical_mapping", {"column": ccol, "mapping": mapping, "set_other": set_other}, [ccol])
                    st.success("Mapping applied.")
                except Exception as e:
                    st.error(f"Error: {e}")

            rare_threshold = st.number_input("Rare category threshold", min_value=1, value=5)
            if st.button("Group Rare Categories"):
                try:
                    new_df = df.copy()
                    counts = new_df[ccol].astype(str).value_counts()
                    rare = counts[counts < rare_threshold].index
                    new_df[ccol] = new_df[ccol].astype(str).where(~new_df[ccol].astype(str).isin(rare), "Other")
                    save_step(new_df, "rare_category_grouping", {"column": ccol, "threshold": int(rare_threshold)}, [ccol])
                    st.success("Rare categories grouped.")
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.button("One Hot Encode"):
                try:
                    new_df = pd.get_dummies(df, columns=[ccol], prefix=ccol)
                    save_step(new_df, "one_hot_encoding", {"column": ccol}, [ccol])
                    st.success("One-hot encoding complete.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No categorical columns found.")

        st.subheader("Outliers")
        if nums:
            ocol = st.selectbox("Numeric column", nums)
            omethod = st.selectbox("Method", ["IQR", "Z-score"])
            oaction = st.selectbox("Action", ["Do nothing", "Remove outlier rows", "Cap / Winsorize"])
            low_q = st.slider("Lower cap quantile", 0.00, 0.20, 0.01, 0.01)
            high_q = st.slider("Upper cap quantile", 0.80, 1.00, 0.99, 0.01)
            mask = outlier_mask(df, ocol, omethod)
            st.write("Detected outliers:", int(mask.fillna(False).sum()))

            if st.button("Apply Outlier Action"):
                try:
                    new_df = df.copy()
                    if oaction == "Remove outlier rows":
                        new_df = new_df[~mask.fillna(False)]
                        msg = f"Removed {len(df) - len(new_df)} rows."
                    elif oaction == "Cap / Winsorize":
                        new_df[ocol] = winsorize(new_df[ocol], low_q, high_q)
                        msg = "Values capped."
                    else:
                        msg = "No changes applied."
                    if oaction != "Do nothing":
                        save_step(new_df, "outlier_handling", {"column": ocol, "method": omethod, "action": oaction}, [ocol])
                    st.success(msg)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.subheader("Scaling")
        if nums:
            scols = st.multiselect("Scale columns", nums)
            smethod = st.selectbox("Method", ["MinMax", "Z-score"])
            if st.button("Apply Scaling"):
                if not scols:
                    st.error("Choose at least one numeric column.")
                else:
                    try:
                        new_df = df.copy()
                        before = pd.DataFrame({
                            "column": scols,
                            "mean_before": [new_df[c].mean() for c in scols],
                            "std_before": [new_df[c].std() for c in scols],
                            "min_before": [new_df[c].min() for c in scols],
                            "max_before": [new_df[c].max() for c in scols]
                        })
                        for c in scols:
                            if smethod == "MinMax":
                                denom = new_df[c].max() - new_df[c].min()
                                new_df[c] = 0 if denom == 0 else (new_df[c] - new_df[c].min()) / denom
                            else:
                                std = new_df[c].std(ddof=0)
                                new_df[c] = 0 if std == 0 else (new_df[c] - new_df[c].mean()) / std
                        after = pd.DataFrame({
                            "column": scols,
                            "mean_after": [new_df[c].mean() for c in scols],
                            "std_after": [new_df[c].std() for c in scols],
                            "min_after": [new_df[c].min() for c in scols],
                            "max_after": [new_df[c].max() for c in scols]
                        })
                        save_step(new_df, "scaling", {"columns": scols, "method": smethod}, scols)
                        st.success("Scaling applied.")
                        st.write("Before")
                        st.dataframe(before, use_container_width=True)
                        st.write("After")
                        st.dataframe(after, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.subheader("Column Operations")
        rcol = st.selectbox("Rename column", all_cols)
        new_name = st.text_input("New column name")
        if st.button("Rename Column"):
            if not new_name.strip():
                st.error("Enter a valid new name.")
            elif new_name in df.columns:
                st.error("That column name already exists.")
            else:
                try:
                    new_df = df.rename(columns={rcol: new_name})
                    save_step(new_df, "rename_column", {"old_name": rcol, "new_name": new_name}, [rcol])
                    st.success("Column renamed.")
                except Exception as e:
                    st.error(f"Error: {e}")

        drop_cols = st.multiselect("Drop columns", all_cols)
        if st.button("Drop Selected Columns"):
            if not drop_cols:
                st.info("Choose at least one column.")
            else:
                try:
                    new_df = df.drop(columns=drop_cols)
                    save_step(new_df, "drop_columns", {"dropped_columns": drop_cols}, drop_cols)
                    st.success("Columns dropped.")
                except Exception as e:
                    st.error(f"Error: {e}")

        if nums:
            ftype = st.selectbox("Formula", ["colA / colB", "log(col)", "colA - mean(colA)"])
            fnew = st.text_input("New formula column name")
            col_a = st.selectbox("Column A", nums)
            col_b = st.selectbox("Column B", nums)
            if st.button("Create Formula Column"):
                if not fnew.strip():
                    st.error("Enter a new column name.")
                else:
                    try:
                        new_df = df.copy()
                        if ftype == "colA / colB":
                            new_df[fnew] = new_df[col_a] / new_df[col_b].replace(0, np.nan)
                        elif ftype == "log(col)":
                            new_df[fnew] = np.where(new_df[col_a] > 0, np.log(new_df[col_a]), np.nan)
                        else:
                            new_df[fnew] = new_df[col_a] - new_df[col_a].mean()
                        save_step(new_df, "create_formula_column", {"formula": ftype, "new_column": fnew, "col_a": col_a, "col_b": col_b}, [fnew])
                        st.success("New column created.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            bcol = st.selectbox("Column to bin", nums)
            bnew = st.text_input("New binned column name")
            bcount = st.slider("Number of bins", 2, 10, 4)
            bmethod = st.selectbox("Binning method", ["Equal-width", "Quantile"])
            if st.button("Apply Binning"):
                if not bnew.strip():
                    st.error("Enter a new column name.")
                else:
                    try:
                        new_df = df.copy()
                        if bmethod == "Equal-width":
                            new_df[bnew] = pd.cut(new_df[bcol], bins=bcount)
                        else:
                            new_df[bnew] = pd.qcut(new_df[bcol], q=bcount, duplicates="drop")
                        save_step(new_df, "bin_numeric_column", {"source_column": bcol, "new_column": bnew, "bins": bcount, "method": bmethod}, [bcol, bnew])
                        st.success("Binning applied.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.subheader("Validation Rules")
        rule = st.selectbox("Choose validation rule", ["Numeric range", "Allowed categories", "Non-null"])

        if rule == "Numeric range" and nums:
            vcol = st.selectbox("Numeric column", nums)
            min_val = st.number_input("Minimum allowed", value=float(df[vcol].min()))
            max_val = st.number_input("Maximum allowed", value=float(df[vcol].max()))
            if st.button("Run Numeric Validation"):
                bad = df[(df[vcol] < min_val) | (df[vcol] > max_val)].copy()
                bad["violation_reason"] = f"{vcol} outside [{min_val}, {max_val}]"
                st.session_state["violations"] = bad
                st.dataframe(bad, use_container_width=True)

        elif rule == "Allowed categories" and cats:
            vcol = st.selectbox("Categorical column", cats)
            allowed = st.text_input("Allowed values comma separated")
            if st.button("Run Category Validation"):
                good = [x.strip() for x in allowed.split(",") if x.strip()]
                bad = df[~df[vcol].astype(str).isin(good)].copy()
                bad["violation_reason"] = f"{vcol} not in allowed list"
                st.session_state["violations"] = bad
                st.dataframe(bad, use_container_width=True)

        elif rule == "Non-null":
            req = st.multiselect("Columns that must not be null", all_cols)
            if st.button("Run Non-null Validation"):
                if not req:
                    st.info("Choose at least one column.")
                else:
                    bad = df[df[req].isna().any(axis=1)].copy()
                    bad["violation_reason"] = f"Null found in {req}"
                    st.session_state["violations"] = bad
                    st.dataframe(bad, use_container_width=True)


elif page == "Visualization":
    st.title("Visualization")

    if has_data():
        df = st.session_state["df"].copy()
        nums = num_cols(df)
        all_cols = list(df.columns)

        st.subheader("Filters")
        fcat = st.selectbox("Category filter column", ["None"] + all_cols)
        if fcat != "None":
            vals = sorted(df[fcat].dropna().astype(str).unique().tolist())
            chosen = st.multiselect("Choose values", vals)
            if chosen:
                df = df[df[fcat].astype(str).isin(chosen)]

        fnum = st.selectbox("Numeric filter column", ["None"] + nums)
        if fnum != "None" and len(df) > 0:
            mn, mx = float(df[fnum].min()), float(df[fnum].max())
            rng = st.slider("Select range", mn, mx, (mn, mx))
            df = df[(df[fnum] >= rng[0]) & (df[fnum] <= rng[1])]

        chart = st.selectbox("Chart", ["Histogram", "Box", "Scatter", "Line", "Bar", "Heatmap"])
        x_col = st.selectbox("X column", ["None"] + all_cols)
        y_col = st.selectbox("Y column", ["None"] + nums)
        group_col = st.selectbox("Group column", ["None"] + all_cols)
        agg = st.selectbox("Aggregation", ["None", "sum", "mean", "count", "median"])
        top_n = st.slider("Top N for bar chart", 3, 30, 10)

        fig, ax = plt.subplots(figsize=(8, 5))

        try:
            if chart == "Histogram":
                if x_col in nums:
                    ax.hist(df[x_col].dropna(), bins=20)
                    ax.set_title(f"Histogram of {x_col}")
                else:
                    st.error("Choose a numeric X column.")

            elif chart == "Box":
                if y_col in nums:
                    if group_col != "None":
                        groups = [g[y_col].dropna().values for _, g in df.groupby(group_col)]
                        labels = [str(k) for k, _ in df.groupby(group_col)]
                        ax.boxplot(groups, tick_labels=labels)
                        ax.tick_params(axis="x", rotation=45)
                    else:
                        ax.boxplot(df[y_col].dropna())
                    ax.set_title(f"Box plot of {y_col}")
                else:
                    st.error("Choose a numeric Y column.")

            elif chart == "Scatter":
                if x_col in nums and y_col in nums:
                    if group_col != "None":
                        for key, g in df.groupby(group_col):
                            ax.scatter(g[x_col], g[y_col], label=str(key), alpha=0.7)
                        ax.legend()
                    else:
                        ax.scatter(df[x_col], df[y_col], alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} vs {x_col}")
                else:
                    st.error("Choose numeric X and Y columns.")

            elif chart == "Line":
                if x_col != "None" and y_col in nums:
                    plot_df = df[[x_col, y_col] + ([group_col] if group_col != "None" else [])].dropna().sort_values(by=x_col)
                    if agg != "None":
                        if group_col != "None":
                            grouped = plot_df.groupby([x_col, group_col], as_index=False)[y_col].agg(agg)
                            for key, g in grouped.groupby(group_col):
                                ax.plot(g[x_col], g[y_col], label=str(key))
                            ax.legend()
                        else:
                            grouped = plot_df.groupby(x_col, as_index=False)[y_col].agg(agg)
                            ax.plot(grouped[x_col], grouped[y_col])
                    else:
                        ax.plot(plot_df[x_col], plot_df[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Line chart of {y_col}")
                else:
                    st.error("Choose X and numeric Y columns.")

            elif chart == "Bar":
                if x_col == "None":
                    st.error("Choose an X column.")
                else:
                    if agg == "None" or y_col == "None":
                        counts = df[x_col].astype(str).value_counts().head(top_n)
                        ax.bar(counts.index, counts.values)
                        ax.set_ylabel("Count")
                    else:
                        grouped = df.groupby(x_col)[y_col].agg(agg).sort_values(ascending=False).head(top_n)
                        ax.bar(grouped.index.astype(str), grouped.values)
                        ax.set_ylabel(f"{agg} of {y_col}")
                    ax.set_title(f"Bar chart of {x_col}")
                    ax.tick_params(axis="x", rotation=45)

            else:
                corr = df.corr(numeric_only=True)
                if corr.empty:
                    st.error("Need numeric columns for heatmap.")
                else:
                    img = ax.imshow(corr, aspect="auto")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(corr.index)))
                    ax.set_yticklabels(corr.index)
                    ax.set_title("Correlation Heatmap")
                    fig.colorbar(img, ax=ax)

            st.pyplot(fig)
        except Exception as e:
            st.error(f"Chart error: {e}")


else:
    st.title("Export & Report")

    if has_data():
        df = st.session_state["df"]

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_data, "cleaned_data.csv", mime="text/csv")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="cleaned_data")
            if not st.session_state["violations"].empty:
                st.session_state["violations"].to_excel(writer, index=False, sheet_name="violations")
        st.download_button(
            "Download Excel",
            excel_buffer.getvalue(),
            "cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("Transformation Log")
        if st.session_state["log"]:
            st.dataframe(pd.DataFrame(st.session_state["log"]), use_container_width=True)
        else:
            st.info("No transformations yet.")

        report = {
            "source_file": st.session_state["file_name"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "steps": st.session_state["log"]
        }
        st.download_button(
            "Download JSON Report",
            json.dumps(report, indent=2, default=str),
            "report.json",
            mime="application/json"
        )

        if not st.session_state["violations"].empty:
            st.subheader("Violations")
            st.dataframe(st.session_state["violations"], use_container_width=True)
            st.download_button(
                "Download Violations CSV",
                st.session_state["violations"].to_csv(index=False).encode("utf-8"),
                "violations.csv",
                mime="text/csv"
            )