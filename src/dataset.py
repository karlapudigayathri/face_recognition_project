# src/dataset.py
from __future__ import annotations
import os, glob, csv, re
from typing import Optional, Tuple, Dict, List
import pandas as pd

# -----------------------------
# File search & safe reading
# -----------------------------
def _find_best_file(patterns: List[str]) -> Optional[str]:
    """
    Among all files matching patterns, return the largest non-empty one.
    Skips 0-byte (empty) files.
    """
    candidates = []
    for pat in patterns:
        for f in glob.glob(pat):
            if os.path.exists(f) and os.path.getsize(f) > 0:
                candidates.append((os.path.getsize(f), f))
    if not candidates:
        return None
    # sort by size, pick largest
    candidates.sort(reverse=True)
    return candidates[0][1]

def _read_table_safely(path: str) -> pd.DataFrame:
    """
    Read CSV/TSV/pipe/semicolon with auto-detected encoding/delimiter.
    Falls back to Excel if needed.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    encodings = ["utf-8", "utf-8-sig", "latin1"]
    delims = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in delims:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    # fallback
    return pd.read_csv(path)

def load_raw(train_path: Optional[str] = None,
             test_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw TRAIN/TEST tables.
    Picks the largest valid file among matches (skips empties).
    """
    if train_path is None:
        train_path = _find_best_file([
            "data/raw/train.csv",
            "data/raw/*train*.csv",
            "/mnt/data/*train*.csv"
        ])
    if test_path is None:
        test_path = _find_best_file([
            "data/raw/test.csv",
            "data/raw/*test*.csv",
            "/mnt/data/*test*.csv"
        ])

    if not train_path:
        raise FileNotFoundError("No usable train file found in data/raw/")
    if not test_path:
        raise FileNotFoundError("No usable test file found in data/raw/")

    print(f"→ Using TRAIN file: {train_path}")
    print(f"→ Using TEST  file: {test_path}")

    train = _read_table_safely(train_path)
    test  = _read_table_safely(test_path)

    print(f"TRAIN shape: {train.shape}")
    print(f"TEST  shape: {test.shape}")
    return train, test

# -----------------------------
# Feature engineering
# -----------------------------
_ISO_DUR_RE = re.compile(
    r"^P(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?"
    r"(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$",
    re.IGNORECASE
)

def _duration_to_seconds(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s.isdigit():
        return float(s)
    m = _ISO_DUR_RE.match(s)
    if m:
        h = int(m.group("h") or 0)
        m_ = int(m.group("m") or 0)
        s_ = int(m.group("s") or 0)
        return float(h*3600 + m_*60 + s_)
    if ":" in s:
        try:
            parts = list(map(int, s.split(":")))
            if len(parts) == 2: return parts[0]*60 + parts[1]
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        except: pass
    return 0.0

def _coerce_int(x) -> int:
    try: return int(float(x))
    except: return 0

_HEADER_ALIASES = {
    "adview": "adview", "adviews": "adview", "ad_views": "adview",
    "vidid": "vidid", "video_id": "vidid", "videoid": "vidid",
    "views": "views", "likes": "likes", "dislikes": "dislikes",
    "comment": "comment", "comments": "comment", "comment_count": "comment",
    "published": "published", "publish_time": "published", "publishdate": "published",
    "duration": "duration", "video_duration": "duration",
    "category": "category", "category_name": "category",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        new_cols.append(_HEADER_ALIASES.get(cl, cl))
    df = df.copy()
    df.columns = new_cols
    return df

def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train = _normalize_columns(train)
    test  = _normalize_columns(test)

    for col in ["views", "likes", "dislikes", "comment"]:
        if col in train: train[col] = train[col].apply(_coerce_int)
        if col in test:  test[col]  = test[col].apply(_coerce_int)

    train["duration_sec"] = train.get("duration", 0).apply(_duration_to_seconds)
    test["duration_sec"]  = test.get("duration", 0).apply(_duration_to_seconds)

    train["published"] = pd.to_datetime(train.get("published"), errors="coerce")
    test["published"]  = pd.to_datetime(test.get("published"), errors="coerce")

    for df in (train, test):
        df["pub_year"]  = df["published"].dt.year.fillna(0).astype(int)
        df["pub_month"] = df["published"].dt.month.fillna(0).astype(int)
        df["pub_day"]   = df["published"].dt.day.fillna(0).astype(int)
        df["pub_dow"]   = df["published"].dt.dayofweek.fillna(0).astype(int)
        df["pub_hour"]  = df["published"].dt.hour.fillna(0).astype(int)

    cats = sorted(set(train["category"].dropna().unique()).union(set(test["category"].dropna().unique())))
    cat2id = {c: i for i, c in enumerate(cats)}
    train["cat_id"] = train["category"].map(cat2id).fillna(-1).astype(int)
    test["cat_id"]  = test["category"].map(cat2id).fillna(-1).astype(int)

    features = [
        "views","likes","dislikes","comment",
        "duration_sec","pub_year","pub_month","pub_day","pub_dow","pub_hour","cat_id"
    ]

    y = train["adview"].astype(float) if "adview" in train.columns else None
    return train[features].astype(float), y, test[features].astype(float), features, cat2id