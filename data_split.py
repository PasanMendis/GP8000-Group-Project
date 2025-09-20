import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ----- Config -----
REQUIRED_COLS = ["Series_Title", "meta_prompt", "overview", "Poster_Image_Path", "Gross", "budget"]
POSTER_CANDIDATES = ["Poster_Image_Path", "poster_path", "PosterPath", "poster", "Poster", "ImagePath"]
BUDGET_CANDIDATES = ["budget", "Budget", "production_budget", "Production_Budget", "Budget_USD", "Budget (USD)"]
RANDOM_STATE = 42
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.20, 0.10
# -------------------

def load_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def choose_poster_col(df: pd.DataFrame) -> str:
    for c in POSTER_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Poster column not found. Tried: {POSTER_CANDIDATES}")

def choose_budget_col(df: pd.DataFrame) -> str | None:
    for c in BUDGET_CANDIDATES:
        if c in df.columns:
            return c
    return None  # will try to parse from text later

_NUM_RE = re.compile(r"[\s,]")

def parse_budget_like(x):
    """
    Robust budget coercion:
      - "$120,000,000" -> 120000000
      - "120M" -> 120e6
      - "0.15B" -> 150e6
      - "120000000" -> 120000000
    Returns float or np.nan.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    # handle suffix M/B
    m = re.match(r"^\$?\s*([0-9]*\.?[0-9]+)\s*([MmBb])?$", s.replace(",", ""))
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        if suf in ("M", "m"):
            return val * 1e6
        if suf in ("B", "b"):
            return val * 1e9
        return val
    # fallback: strip non-numeric formatting like commas, spaces, $
    s2 = re.sub(_NUM_RE, "", s.replace("$", ""))
    try:
        return float(s2)
    except Exception:
        return np.nan

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def make_key(df: pd.DataFrame) -> pd.Series:
    """Global uniqueness key: Series_Title + Poster_Image_Path (case-insensitive, trimmed)."""
    title = df["Series_Title"].astype(str).str.strip().str.lower()
    poster = df["Poster_Image_Path"].astype(str).str.strip().str.lower()
    return title + "||" + poster

def sanitize_existing(train_p: Path, val_p: Path, test_p: Path):
    """
    Load existing train/val/test (if any), ensure REQUIRED_COLS exist,
    drop internal duplicates per file, then enforce global uniqueness
    with priority: train > val > test.
    """
    def maybe_load(p):
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=REQUIRED_COLS)

    tr = maybe_load(train_p)
    va = maybe_load(val_p)
    te = maybe_load(test_p)

    # Ensure all REQUIRED_COLS are present
    for df in (tr, va, te):
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)

    # Coerce types where applicable
    for df in (tr, va, te):
        df["Gross"] = df["Gross"].apply(coerce_float)
        df["budget"] = df["budget"].apply(parse_budget_like)

    # Drop internal duplicates by key
    for df in (tr, va, te):
        df["_key"] = make_key(df)
        df.drop_duplicates(subset=["_key"], keep="first", inplace=True)
        df.drop(columns=["_key"], inplace=True)

    # Enforce cross-file uniqueness (keep train, then filter val/test)
    tr["_key"] = make_key(tr); va["_key"] = make_key(va); te["_key"] = make_key(te)
    train_keys = set(tr["_key"].tolist())
    va = va[~va["_key"].isin(train_keys)].copy()
    val_keys = set(va["_key"].tolist())
    te = te[~te["_key"].isin(train_keys.union(val_keys))].copy()
    for df in (tr, va, te):
        df.drop(columns=["_key"], inplace=True)

    return tr, va, te

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_and_append_unique.py <input_file.csv/xlsx>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load input
    df = load_table(in_path)

    # Map poster -> Poster_Image_Path
    poster_col = choose_poster_col(df)
    if poster_col != "Poster_Image_Path":
        df["Poster_Image_Path"] = df[poster_col]

    # Map budget -> budget (or try to parse later from meta_prompt if absent)
    budget_col = choose_budget_col(df)
    if budget_col is not None:
        df["budget"] = df[budget_col].apply(parse_budget_like)
    else:
        # try weak parse from meta_prompt, else NaN
        if "meta_prompt" in df.columns:
            df["budget"] = df["meta_prompt"].apply(parse_budget_like)
        else:
            df["budget"] = np.nan

    # Check core required columns are present
    missing_cols = [c for c in ["Series_Title","meta_prompt","overview","Poster_Image_Path","Gross"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Make sure your file has them.")

    # Clean target and drop rows with required missing values
    df["Gross"] = df["Gross"].apply(coerce_float)
    df = df.dropna(subset=["Series_Title","meta_prompt","overview","Poster_Image_Path","Gross"]).copy()

    # Ensure budget exists (may be NaN; that’s fine — training code will handle)
    df["budget"] = df["budget"].apply(parse_budget_like)

    # Make unique within incoming batch
    df["_key"] = make_key(df)
    df.drop_duplicates(subset=["_key"], keep="first", inplace=True)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Keep only required columns for splitting
    df_req = df[REQUIRED_COLS + ["_key"]].copy()

    # Load & sanitize existing outputs
    train_p, val_p, test_p = Path("train.csv"), Path("val.csv"), Path("test.csv")
    tr_old, va_old, te_old = sanitize_existing(train_p, val_p, test_p)

    # Exclude any rows that already exist in any existing file
    existing_keys = set()
    if len(tr_old):
        tr_old["_key"] = make_key(tr_old); existing_keys |= set(tr_old["_key"].tolist())
    if len(va_old):
        va_old["_key"] = make_key(va_old); existing_keys |= set(va_old["_key"].tolist())
    if len(te_old):
        te_old["_key"] = make_key(te_old); existing_keys |= set(te_old["_key"].tolist())

    df_new = df_req[~df_req["_key"].isin(existing_keys)].copy()

    # If nothing new, just rewrite sanitized existing and exit
    if df_new.empty:
        for dfx, pth in ((tr_old, train_p), (va_old, val_p), (te_old, test_p)):
            if "_key" in dfx.columns: dfx = dfx.drop(columns=["_key"])
            save_csv(dfx[REQUIRED_COLS], pth)
        print("No new unique rows to add. Existing train/val/test were sanitized for duplicates.")
        return

    # Split df_new into train/val/test
    df_rest, df_test = train_test_split(df_new, test_size=TEST_PCT, random_state=RANDOM_STATE)
    val_frac_of_rest = VAL_PCT / (1 - TEST_PCT)
    df_train, df_val = train_test_split(df_rest, test_size=val_frac_of_rest, random_state=RANDOM_STATE)

    # Remove helper key before saving/appending
    for dfx in (df_train, df_val, df_test):
        if "_key" in dfx.columns:
            dfx.drop(columns=["_key"], inplace=True)

    # Append to sanitized existing sets (preserve REQUIRED_COLS order)
    tr_final = pd.concat([tr_old[REQUIRED_COLS], df_train[REQUIRED_COLS]], ignore_index=True)
    va_final = pd.concat([va_old[REQUIRED_COLS], df_val[REQUIRED_COLS]], ignore_index=True)
    te_final = pd.concat([te_old[REQUIRED_COLS], df_test[REQUIRED_COLS]], ignore_index=True)

    # Final safety pass: enforce global uniqueness (priority: train > val > test)
    def enforce_global_unique(tr, va, te):
        tr["_key"] = make_key(tr); va["_key"] = make_key(va); te["_key"] = make_key(te)
        tr = tr.drop_duplicates("_key", keep="first")
        va = va[~va["_key"].isin(set(tr["_key"]))].drop_duplicates("_key", keep="first")
        te = te[~te["_key"].isin(set(tr["_key"]).union(set(va["_key"])))].drop_duplicates("_key", keep="first")
        for dfx in (tr, va, te):
            dfx.drop(columns=["_key"], inplace=True)
        return tr, va, te

    tr_final, va_final, te_final = enforce_global_unique(tr_final, va_final, te_final)

    # Save
    save_csv(tr_final[REQUIRED_COLS], train_p)
    save_csv(va_final[REQUIRED_COLS], val_p)
    save_csv(te_final[REQUIRED_COLS], test_p)

    print(f"✅ Done. Appended new unique rows and saved:")
    print(f"   train.csv: {len(tr_final)} rows")
    print(f"   val.csv:   {len(va_final)} rows")
    print(f"   test.csv:  {len(te_final)} rows")

if __name__ == "__main__":
    main()