# -*- coding: utf-8 -*-
"""
onlypicture 分群 + wavelet 小波轉換（四群 → 四張圖；dp001 為基底 / review_sn 對回 user_sn）
- 優先讀 BASE_DIR/clusters_km_xgb.csv；沒有就內嵌 onlypicture 分群（以 dp001 活躍者為基底 ≈ 95 人）
- 讀入 dp001/2/3/4 行為檔，合併成事件表（ts、user_sn、weight）
- 依分群重採樣 → 逐群畫 CWT (Morlet) Scalogram，輸出四張 PNG

需要套件：pandas numpy matplotlib scikit-learn xgboost PyWavelets（或 scipy）
"""

import os, re, sys, math, csv, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 路徑與參數（請按需修改） =========
BASE_DIR = Path(r"D:\專題\教育大數據比賽")
RESAMPLE_RULE = "D"           # 'D' (日) 或 'H' (小時)
GROUP_AGG = "sum"             # 'sum'（總強度）或 'mean'（人均）
LABEL_COL = "xgb_cluster_name"
EXPECTED_GROUPS = ["Advanced","Diligent","Persistent","Indifferent"]
EXPORT_CLEAN = True
# ========================================

# --- CWT 套件：優先 PyWavelets，否則 SciPy ---
USE_PYWT = True
try:
    import pywt
except Exception:
    USE_PYWT = False
    from scipy.signal import cwt, morlet2  # type: ignore

# --- Wavelet 參數 ---
SCALES = np.geomspace(2, 256, 128)      # PyWavelets 用
PERIODS_FOR_SCIPY = np.geomspace(2, 256, 128)  # SciPy fallback 用（單位=天）

# ----------------- 常用工具 -----------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin1"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, engine="python", encoding=enc, sep=sep,
                                 quotechar='"', doublequote=True, escapechar="\\",
                                 on_bad_lines="warn")
                if df.shape[1] > 1:
                    print(f"[OK] {path.name}: encoding={enc}, sep={repr(sep)} (warn)")
                    return df
            except Exception as e:
                last_err = e
            try:
                df = pd.read_csv(path, engine="python", encoding=enc, sep=sep,
                                 quotechar='"', doublequote=True, escapechar="\\",
                                 on_bad_lines="skip")
                if df.shape[1] > 1:
                    print(f"[OK] {path.name}: encoding={enc}, sep={repr(sep)} (skip)")
                    return df
            except Exception as e2:
                last_err = e2
    raise RuntimeError(f"[FAIL] 讀檔失敗：{path}\n最後錯誤：{last_err}")

def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def to_datetime_smart(sr: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(sr):
        med = sr.dropna().astype(float).median() if sr.dropna().size else np.nan
        if pd.notna(med):
            if med > 1e12: return pd.to_datetime(sr, unit="ms", errors="coerce")
            if med > 1e9:  return pd.to_datetime(sr, unit="s",  errors="coerce")
    return pd.to_datetime(sr, errors="coerce")

def export_clean(df: pd.DataFrame, src: Path):
    if not EXPORT_CLEAN: return
    out_dir = BASE_DIR / "clean"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / src.name.replace(src.suffix, "_clean.csv")
    try:
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[CLEAN] 輸出：{out}")
    except Exception as e:
        print(f"[WARN] 無法輸出清理檔：{e}")

def parse_iso8601_duration(s):
    if pd.isna(s): return np.nan
    if not isinstance(s, str): s = str(s)
    m = re.fullmatch(r'P(?:\d+Y)?(?:\d+M)?(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', s)
    if not m: return np.nan
    h = int(m.group(1) or 0); m_ = int(m.group(2) or 0); s_ = float(m.group(3) or 0)
    return h*3600 + m_*60 + s_

# merge key 正規化：統一成乾淨字串（保留前導 0，去掉尾端 .0）
def norm_key_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r'\.0$', '', regex=True)
    s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'null': np.nan})
    return s

# 自動找 user/time 欄位
USER_CANDIDATES = ["user_sn","user_id","userid","account","uid","username","sno","sid","student_id"]
TIME_KEYWORDS = ["view_time","finish_time","action_time","start_time","end_time",
                 "timestamp","date","time","created","modified","game_time","last_modified"]

def pick_user_col(df: pd.DataFrame) -> str|None:
    for c in df.columns:
        lc = c.lower()
        if lc in USER_CANDIDATES or any(k in lc for k in USER_CANDIDATES):
            return c
    return None

def pick_time_col(df: pd.DataFrame, preferred: list[str]|None=None) -> str|None:
    if preferred:
        for c in preferred:
            if c in df.columns: return c
    for c in df.columns:
        if any(k in c.lower() for k in TIME_KEYWORDS):
            return c
    return None

# --------------- 各資料表 Loader（事件用） ---------------
def load_dp001_review(base: Path) -> pd.DataFrame:
    p = base / "dp001_review.csv"
    if not p.exists():
        print("[MISS] dp001_review.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype","review_sn"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df)
    if not ucol:
        print("[WARN] review: 找不到 user 欄位")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype","review_sn"])
    df[ucol] = norm_id(df[ucol])
    if "review_sn" in df.columns: df["review_sn"] = norm_key_str(df["review_sn"])
    else: df["review_sn"] = np.nan

    tcol = pick_time_col(df, preferred=["end_time","start_time"])
    if not tcol:
        print("[WARN] review: 找不到時間欄位")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype","review_sn"])
    df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])

    w = None
    if set(["start_time","end_time"]).issubset(df.columns):
        df["start_time"] = to_datetime_smart(df["start_time"])
        df["end_time"] = to_datetime_smart(df["end_time"])
        w = (df["end_time"] - df["start_time"]).dt.total_seconds()
    if w is None or (isinstance(w,pd.Series) and w.isna().all()):
        if set(["video_len","finish_rate"]).issubset(df.columns):
            w = pd.to_numeric(df["video_len"], errors="coerce") * pd.to_numeric(df["finish_rate"], errors="coerce")
        else:
            w = 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (w if isinstance(w,(int,float)) else w.fillna(1.0)),
        "channel": "review",
        "subtype": df["subject_name"] if "subject_name" in df.columns else "NA",
        "review_sn": df["review_sn"]
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp001_review_plus(base: Path, review_map: pd.DataFrame) -> pd.DataFrame:
    p = base / "dp001_review_plus.csv"
    if not p.exists():
        print("[MISS] dp001_review_plus.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    tcol = pick_time_col(df, preferred=["view_time"])
    if not tcol:
        print("[WARN] review_plus: 找不到時間欄位")
        return pd.DataFrame(columns=[])
    df[tcol] = to_datetime_smart(df[tcol]); df = df.dropna(subset=[tcol])

    if "review_sn" in df.columns:
        df["review_sn"] = norm_key_str(df["review_sn"])
        # 用 review_sn 對回 user_sn
        df = df.merge(review_map, on="review_sn", how="left")

    if "user_sn" not in df.columns: df["user_sn"] = np.nan
    miss = df["user_sn"].isna()
    if miss.any():
        if "review_sn" in df.columns:
            df.loc[miss,"user_sn"] = "R:" + df.loc[miss,"review_sn"].astype(str)
        else:
            df.loc[miss,"user_sn"] = "REVLESS:" + df.index[miss].astype(str)

    out = pd.DataFrame({
        "user_sn": norm_id(df["user_sn"]),
        "ts": df[tcol],
        "weight": 1.0,
        "channel": "review_plus",
        "subtype": df["view_action"] if "view_action" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp001_exam(base: Path, review_map: pd.DataFrame) -> pd.DataFrame:
    p = base / "dp001_exam.csv"
    if not p.exists():
        print("[MISS] dp001_exam.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    tcol = pick_time_col(df, preferred=["finish_time"])
    if not tcol:
        print("[WARN] exam_v1: 找不到時間欄位")
        return pd.DataFrame(columns=[])
    df[tcol] = to_datetime_smart(df[tcol]); df = df.dropna(subset=[tcol])

    if "review_sn" in df.columns:
        df["review_sn"] = norm_key_str(df["review_sn"])
        df = df.merge(review_map, on="review_sn", how="left")

    if "user_sn" not in df.columns: df["user_sn"] = "EXAMLESS:" + df.index.astype(str)

    out = pd.DataFrame({
        "user_sn": norm_id(df["user_sn"]),
        "ts": df[tcol],
        "weight": pd.to_numeric(df.get("ans_time", 1.0), errors="coerce").fillna(1.0),
        "channel": "exam_v1",
        "subtype": df["binary_res"] if "binary_res" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp001_prac(base: Path) -> pd.DataFrame:
    p = base / "dp001_prac.csv"
    if not p.exists():
        print("[MISS] dp001_prac.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df)
    tcol = pick_time_col(df, preferred=["date"])
    if not ucol or not tcol:
        print("[WARN] prac: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])

    if "during_time" in df.columns and df["during_time"].notna().any():
        w = pd.to_numeric(df["during_time"], errors="coerce")
    elif "items_ans_time" in df.columns:
        w = pd.to_numeric(df["items_ans_time"], errors="coerce")
    else:
        w = 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (w if isinstance(w,(int,float)) else w.fillna(1.0)),
        "channel": "practice",
        "subtype": df["subject_name"] if "subject_name" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp002_exam(base: Path) -> pd.DataFrame:
    p = base / "dp002_exam.csv"
    if not p.exists():
        print("[MISS] dp002_exam.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df); tcol = pick_time_col(df, preferred=["action_time"])
    if not ucol or not tcol:
        print("[WARN] dp002_exam: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])
    dur = df["result_duration"].apply(parse_iso8601_duration) if "result_duration" in df.columns else 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (dur if isinstance(dur,(int,float)) else pd.to_numeric(dur, errors="coerce").fillna(1.0)),
        "channel": "exam_v2",
        "subtype": df["object_type"] if "object_type" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp003_word(base: Path) -> pd.DataFrame:
    p = base / "dp003_word.csv"
    if not p.exists():
        print("[MISS] dp003_word.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df)
    tcol = pick_time_col(df, preferred=["finish_timestamp","start_timestamp","last_modified"])
    if not ucol or not tcol:
        print("[WARN] word: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])

    if "total_millisecond" in df.columns:
        w = pd.to_numeric(df["total_millisecond"], errors="coerce").div(1000)
    else:
        w = 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (w if isinstance(w,(int,float)) else w.fillna(1.0)),
        "channel": "word_game",
        "subtype": df["section_id"] if "section_id" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp003_math(base: Path) -> pd.DataFrame:
    p = base / "dp003_math.csv"
    if not p.exists():
        print("[MISS] dp003_math.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df); tcol = pick_time_col(df, preferred=["game_time","last_modified"])
    if not ucol or not tcol:
        print("[WARN] math: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])
    w = pd.to_numeric(df.get("game_grade",1.0), errors="coerce").fillna(1.0)

    out = pd.DataFrame({
        "user_sn": df[ucol], "ts": df[tcol], "weight": w,
        "channel": "math_game",
        "subtype": df["unit_name"] if "unit_name" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp004_interaction(base: Path) -> pd.DataFrame:
    p = base / "dp004_interaction.csv"
    if not p.exists():
        print("[MISS] dp004_interaction.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df); tcol = pick_time_col(df, preferred=["timestamp"])
    if not ucol or not tcol:
        print("[WARN] interaction: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])
    dur = df["result_duration"].apply(parse_iso8601_duration) if "result_duration" in df.columns else 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (dur if isinstance(dur,(int,float)) else pd.to_numeric(dur, errors="coerce").fillna(1.0)),
        "channel": "interaction",
        "subtype": df["verb_name"] if "verb_name" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp004_video(base: Path) -> pd.DataFrame:
    p = base / "dp004_video.csv"
    if not p.exists():
        print("[MISS] dp004_video.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df); tcol = pick_time_col(df, preferred=["timestamp"])
    if not ucol or not tcol:
        print("[WARN] video: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])
    dur = df["result_duration"].apply(parse_iso8601_duration) if "result_duration" in df.columns else 1.0

    out = pd.DataFrame({
        "user_sn": df[ucol],
        "ts": df[tcol],
        "weight": (dur if isinstance(dur,(int,float)) else pd.to_numeric(dur, errors="coerce").fillna(1.0)),
        "channel": "video",
        "subtype": df["object_name"] if "object_name" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

def load_dp004_webpage(base: Path) -> pd.DataFrame:
    p = base / "dp004_webpage.csv"
    if not p.exists():
        print("[MISS] dp004_webpage.csv 不存在")
        return pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    df = read_csv_robust(p)
    ucol = pick_user_col(df); tcol = pick_time_col(df, preferred=["timestamp"])
    if not ucol or not tcol:
        print("[WARN] webpage: 缺 user 或時間欄位")
        return pd.DataFrame(columns=[])
    df[ucol] = norm_id(df[ucol]); df[tcol] = to_datetime_smart(df[tcol])
    df = df.dropna(subset=[ucol,tcol])

    out = pd.DataFrame({
        "user_sn": df[ucol], "ts": df[tcol],
        "weight": 1.0, "channel": "webpage",
        "subtype": df["object_name"] if "object_name" in df.columns else "NA"
    }).dropna(subset=["user_sn","ts"])
    export_clean(out,p)
    return out

# --------------- 建事件表 ---------------
def build_events_table(base: Path) -> pd.DataFrame:
    print(f"\n[INFO] 讀檔資料夾：{base}")
    review = load_dp001_review(base)
    review_map = review[["review_sn","user_sn"]].dropna().drop_duplicates() if not review.empty else pd.DataFrame(columns=["review_sn","user_sn"])
    frames = [
        review,
        load_dp001_review_plus(base, review_map),
        load_dp001_exam(base, review_map),
        load_dp001_prac(base),
        load_dp002_exam(base),
        load_dp003_word(base),
        load_dp003_math(base),
        load_dp004_interaction(base),
        load_dp004_video(base),
        load_dp004_webpage(base),
    ]
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["user_sn","ts","weight","channel","subtype"])
    if events.empty:
        print("[SUM] 合併後事件數：0"); return events
    events = events.dropna(subset=["user_sn","ts"])
    events["ts"] = pd.to_datetime(events["ts"], errors="coerce"); events = events.dropna(subset=["ts"])
    events["weight"] = pd.to_numeric(events["weight"], errors="coerce").fillna(0); events.loc[events["weight"]<0,"weight"]=0
    print(f"[SUM] 合併後事件數：{len(events)}；期間：{events['ts'].min()} ~ {events['ts'].max()}")
    return events

# =========================
# 🔽🔽 onlypicture 分群（內嵌，dp001 基底）🔽🔽
# =========================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier

EXPECTED_PATTERNS = {
    "dp001_exam.csv":         r"dp001.*exam.*\.(csv|xlsx)$",
    "dp001_prac.csv":         r"dp001.*prac.*\.(csv|xlsx)$",
    "dp001_review.csv":       r"dp001.*review(?!.*plus).*\.((csv)|(xlsx))$",
    "dp001_review_plus.csv":  r"dp001.*review.*plus.*\.(csv|xlsx)$",
    "dp002_exam.csv":         r"dp002.*exam.*\.(csv|xlsx)$",
    "dp003_math.csv":         r"dp003.*math.*\.(csv|xlsx)$",
    "dp003_word.csv":         r"dp003.*word.*\.(csv|xlsx)$",
    "dp004_interaction.csv":  r"dp004.*interact.*\.(csv|xlsx)$",
    "dp004_video.csv":        r"dp004.*video.*\.(csv|xlsx)$",
    "dp004_webpage.csv":      r"dp004.*web[_ ]?page.*\.(csv|xlsx)$",
    "user_data.csv":          r"user.*data.*\.(csv|xlsx)$",
}

def robust_read_onlypicture(fpath):
    if fpath.lower().endswith(".xlsx"):
        try:
            return pd.read_excel(fpath, engine="openpyxl"), "excel"
        except Exception:
            return pd.read_excel(fpath), "excel"
    tries = [
        dict(encoding="utf-8-sig", engine="python", on_bad_lines="skip"),
        dict(encoding="utf-8", engine="python", on_bad_lines="skip"),
        dict(encoding="latin1", engine="python", on_bad_lines="skip"),
        dict(encoding="cp950", engine="python", on_bad_lines="skip", encoding_errors="ignore"),
    ]
    last=None
    for opt in tries:
        try:
            return pd.read_csv(fpath, **opt), f"csv:{opt}"
        except Exception as e:
            last=e
    raise RuntimeError(f"讀檔失敗：{fpath} ({last})")

def onlypicture_load_csvs(data_dir: str):
    abs_dir = os.path.abspath(data_dir)
    print(f"[INFO] Scanning for clustering: {abs_dir}\n")
    def first_match(root, pat):
        creg = re.compile(pat, re.IGNORECASE)
        for r, _, files in os.walk(root):
            for fn in files:
                if creg.search(fn):
                    return os.path.join(r, fn)
        return None
    by_name = {}
    for canon, pat in EXPECTED_PATTERNS.items():
        fpath = first_match(abs_dir, pat)
        if not fpath: continue
        try:
            df, how = robust_read_onlypicture(fpath)
            for c in list(df.columns):
                if c.strip().lower()=="user_sn":
                    df.rename(columns={c:"user_sn"}, inplace=True)
            if "user_sn" in df.columns:
                df["user_sn"] = norm_id(df["user_sn"])
            by_name[canon] = df
            print(f"[FOUND] {canon}  <=  {fpath}  ({how})  行×欄: {df.shape}")
        except Exception as e:
            print(f"[WARN] 讀取失敗 {canon} : {e}")
    print("\n[INFO] 讀到檔案：", ", ".join(by_name.keys()) or "(無)")
    return by_name

# ⬇︎ 基底使用者：dp001 活躍者 + review_sn 對回 user_sn（預期 ~95 人）
STRICT_DP001_BASE = True

def onlypicture_base_users_dp001(by_name: dict) -> pd.Series:
    """
    以 dp001 為基底，且使用 review_sn 對回 user_sn。
    嚴格模式（預設）：base = review ∩ (review_plus→user) ∩ (exam→user)
    寬鬆模式：      base = review ∪ (review_plus→user) ∪ (exam→user)
    """
    # A) dp001_review（直接有 user_sn）
    review_df = by_name.get("dp001_review.csv")
    set_review = set()
    rv = None
    if review_df is not None and "user_sn" in review_df.columns:
        tmp = review_df.copy()
        tmp["user_sn"] = norm_id(tmp["user_sn"])
        if "review_sn" in tmp.columns:
            tmp["review_sn"] = norm_key_str(tmp["review_sn"])
            rv = tmp[["review_sn","user_sn"]].dropna().drop_duplicates()
        set_review = set(tmp["user_sn"].dropna().unique())

    # B) dp001_review_plus 透過 review_sn -> user_sn
    set_plus = set()
    plus_df = by_name.get("dp001_review_plus.csv")
    if rv is not None and plus_df is not None and "review_sn" in plus_df.columns:
        t = plus_df.copy()
        t["review_sn"] = norm_key_str(t["review_sn"])
        t = t.merge(rv, on="review_sn", how="left").dropna(subset=["user_sn"])
        set_plus = set(norm_id(t["user_sn"]).dropna().unique())

    # C) dp001_exam 透過 review_sn -> user_sn
    set_exam = set()
    exam1_df = by_name.get("dp001_exam.csv")
    if rv is not None and exam1_df is not None and "review_sn" in exam1_df.columns:
        t = exam1_df.copy()
        t["review_sn"] = norm_key_str(t["review_sn"])
        t = t.merge(rv, on="review_sn", how="left").dropna(subset=["user_sn"])
        set_exam = set(norm_id(t["user_sn"]).dropna().unique())

    # ---- 組合規則 ----
    pools = [s for s in [set_review, set_plus, set_exam] if len(s) > 0]
    if not pools:
        # 若三個都讀不到，就退回 user_data.csv 或 union(all)
        print("[BASE] 無 dp001 基底可用，退回 user_data 或 union(all)")
        if "user_data.csv" in by_name and "user_sn" in by_name["user_data.csv"].columns:
            return pd.Series(norm_id(by_name["user_data.csv"]["user_sn"]).dropna().unique())
        all_user = []
        for name, df in by_name.items():
            if "user_sn" in df.columns:
                all_user.append(norm_id(df["user_sn"]).dropna())
        return pd.Series(pd.concat(all_user).unique()) if all_user else pd.Series([], dtype=object)

    if STRICT_DP001_BASE and len(pools) >= 2:
        base_set = pools[0].copy()
        for s in pools[1:]:
            base_set &= s
        base_mode = "交集 (strict)"
    else:
        base_set = set().union(*pools)
        base_mode = "聯集 (loose)"

    base = pd.Series(sorted(base_set))
    print(f"[BASE] dp001 基底（{base_mode}）：review={len(set_review)}, plus={len(set_plus)}, exam={len(set_exam)} → 基底={len(base)} 人")
    return base

def onlypicture_build_static_features(by_name: dict) -> pd.DataFrame:
    print("\n[INFO] 開始特徵工程...")
    users = onlypicture_base_users_dp001(by_name)   # ✅ 約 95 人
    base = pd.DataFrame({"user_sn": norm_id(users)})

    print(f"[CHECK] 各檔案內的 user_sn 去重數：")
    for k, v in by_name.items():
        if "user_sn" in v.columns:
            print(f"  - {k:24s}: {norm_id(v['user_sn']).dropna().nunique():4d}")

    # A) user_data（三科分數）
    user_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "user_data.csv" in by_name:
        t = by_name["user_data.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        cols = [c for c in ["user_sn","chinese_score","math_score","english_score"] if c in t.columns]
        t = t[cols].groupby("user_sn", as_index=False).mean()
        user_feat = user_feat.merge(t, on="user_sn", how="left")
    for c in ["chinese_score","math_score","english_score"]:
        if c not in user_feat.columns: user_feat[c] = 0.0

    # B) dp001_review
    review_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    review_df = by_name.get("dp001_review.csv")
    if review_df is not None:
        tmp = review_df.copy()
        if "user_sn" in tmp.columns: tmp["user_sn"] = norm_id(tmp["user_sn"])
        for c in ["video_len","start_timestamp","end_timestamp","finish_rate"]:
            if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        agg = tmp.groupby("user_sn").agg(
            review_session_count=("review_sn","nunique") if "review_sn" in tmp.columns else ("user_sn","count"),
            video_finish_rate_mean=("finish_rate", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "finish_rate" in tmp.columns else ("user_sn","size"),
            video_len_mean=("video_len", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "video_len" in tmp.columns else ("user_sn","size"),
            video_len_median=("video_len", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().median()) if s is not None and not pd.to_numeric(s, errors="coerce").dropna().empty else 0.0)
        ).reset_index()
        def row_watch_ratio(r):
            try:
                st, et, L = float(r.get("start_timestamp", np.nan)), float(r.get("end_timestamp", np.nan)), float(r.get("video_len", np.nan))
                if any(math.isnan(x) for x in [st, et, L]) or L<=0: return np.nan
                seg = max(0.0, min(et - st, L)); return seg / L
            except: return np.nan
        tmp["watch_ratio_est"] = tmp.apply(row_watch_ratio, axis=1)
        wr = tmp.groupby("user_sn")["watch_ratio_est"].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0).rename("watch_ratio_mean").reset_index()
        review_feat = review_feat.merge(agg, on="user_sn", how="left").merge(wr, on="user_sn", how="left")
    for c in ["review_session_count","video_finish_rate_mean","video_len_mean","video_len_median","watch_ratio_mean"]:
        if c not in review_feat.columns: review_feat[c] = 0.0

    # C) dp001_review_plus（review_sn -> user_sn）
    plus_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    plus_df = by_name.get("dp001_review_plus.csv")
    if (review_df is not None) and (plus_df is not None) and {"review_sn","view_action"}.issubset(plus_df.columns):
        rv = review_df[["review_sn","user_sn"]].dropna().copy()
        rv["user_sn"] = norm_id(rv["user_sn"])
        rv["review_sn"] = norm_key_str(rv["review_sn"])
        merged = plus_df.copy()
        merged["review_sn"] = norm_key_str(merged["review_sn"])
        merged = merged.merge(rv, on="review_sn", how="left").dropna(subset=["user_sn"])
        counts = merged.groupby(["user_sn","view_action"]).size().unstack(fill_value=0)
        for act in ["dragleft","dragright","paused","play","end","speedup","slowdown","note","normal"]:
            if act not in counts.columns: counts[act] = 0
        counts = counts.reset_index().rename(columns={
            "dragleft":"SkippingBackward_freq","dragright":"SkippingForward_freq","paused":"Pause_freq",
            "play":"Play_freq","end":"End_freq","speedup":"Speedup_freq","slowdown":"Slowdown_freq","note":"Note_freq"
        })
        counts["action_total_count"] = counts.drop(columns=["user_sn"]).sum(axis=1)
        counts["action_variety"] = (counts.drop(columns=["user_sn"])>0).sum(axis=1)
        total = counts["action_total_count"].replace(0, 1.0)
        counts["Seek_ratio"]  = (counts["SkippingBackward_freq"] + counts["SkippingForward_freq"]) / total
        counts["Pause_ratio"] = counts["Pause_freq"] / total
        counts["Speed_change_ratio"] = (counts["Speedup_freq"] + counts["Slowdown_freq"]) / total
        plus_feat = counts
    else:
        cols0 = ["SkippingBackward_freq","SkippingForward_freq","Pause_freq","Play_freq","End_freq","Speedup_freq","Slowdown_freq","Note_freq",
                 "action_total_count","action_variety","Seek_ratio","Pause_ratio","Speed_change_ratio"]
        plus_feat = pd.DataFrame({"user_sn": base["user_sn"], **{c:0.0 for c in cols0}})

    # D) dp001_prac
    prac_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp001_prac.csv" in by_name:
        t = by_name["dp001_prac.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        for col in ["score_rate","during_time"]:
            if col in t.columns: t[col] = pd.to_numeric(t[col], errors="coerce")
        if "items_ans_time" in t.columns:
            def mean_ms(s):
                if pd.isna(s): return np.nan
                parts = [p for p in str(s).split("@XX@") if p.strip()!=""]
                arr = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
                return float(arr.mean()) if not arr.empty else np.nan
            t["items_ans_time_mean_ms_row"] = t["items_ans_time"].map(mean_ms)
        else:
            t["items_ans_time_mean_ms_row"] = np.nan
        agg = t.groupby("user_sn").agg(
            prac_score_rate_mean=("score_rate", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0),
            prac_during_time_mean_sec=("during_time", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0),
            prac_item_time_mean_ms=("items_ans_time_mean_ms_row", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0),
            prac_count=("prac_sn","count") if "prac_sn" in t.columns else ("user_sn","count")
        ).reset_index()
        prac_feat = prac_feat.merge(agg, on="user_sn", how="left")
    for c in ["prac_score_rate_mean","prac_during_time_mean_sec","prac_item_time_mean_ms","prac_count"]:
        if c not in prac_feat.columns: prac_feat[c] = 0.0

    # E) dp001_exam（review_sn -> user_sn）
    exam1_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp001_exam.csv" in by_name and review_df is not None:
        t = by_name["dp001_exam.csv"].copy()
        rv = review_df[["review_sn","user_sn"]].dropna().copy()
        rv["user_sn"] = norm_id(rv["user_sn"])
        rv["review_sn"] = norm_key_str(rv["review_sn"])
        t["review_sn"] = norm_key_str(t["review_sn"]) if "review_sn" in t.columns else np.nan
        t = t.merge(rv, on="review_sn", how="left").dropna(subset=["user_sn"])
        t["user_sn"] = norm_id(t["user_sn"])
        if "ans_time" in t.columns: t["ans_time"] = pd.to_numeric(t["ans_time"], errors="coerce")
        if "binary_res" in t.columns: t["binary_res_num"] = pd.to_numeric(t["binary_res"], errors="coerce")
        agg = t.groupby("user_sn").agg(
            exam_ans_time_mean=("ans_time", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "ans_time" in t.columns else ("user_sn","size"),
            exam_accuracy=("binary_res_num", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "binary_res_num" in t.columns else ("user_sn","size"),
            exam_count=("question_sn","count") if "question_sn" in t.columns else ("user_sn","count")
        ).reset_index()
        exam1_feat = exam1_feat.merge(agg, on="user_sn", how="left")
    for c in ["exam_ans_time_mean","exam_accuracy","exam_count"]:
        if c not in exam1_feat.columns: exam1_feat[c] = 0.0

    # F) dp002_exam
    exam2_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp002_exam.csv" in by_name:
        t = by_name["dp002_exam.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        if "result_duration" in t.columns:
            t["duration_sec"] = t["result_duration"].map(lambda x: parse_iso8601_duration(x))
        if "result_success" in t.columns:
            def to01(v):
                s = str(v).strip().lower()
                if s in ("true","1","y","yes"): return 1.0
                if s in ("false","0","n","no"): return 0.0
                return np.nan
            t["success01"] = t["result_success"].map(to01)
        agg = t.groupby("user_sn").agg(
            dp002_success_rate=("success01", lambda s: s.dropna().mean() if s is not None else 0.0) if "success01" in t.columns else ("user_sn","size"),
            dp002_duration_mean_sec=("duration_sec", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "duration_sec" in t.columns else ("user_sn","size"),
            dp002_mission_count=("mission_id","nunique") if "mission_id" in t.columns else ("user_sn","count")
        ).reset_index()
        exam2_feat = exam2_feat.merge(agg, on="user_sn", how="left")
    for c in ["dp002_success_rate","dp002_duration_mean_sec","dp002_mission_count"]:
        if c not in exam2_feat.columns: exam2_feat[c] = 0.0

    # G) dp003_word —— 欄位缺也能跑
    word_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp003_word.csv" in by_name:
        t = by_name["dp003_word.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        need_num_cols = ["isCorrect", "wrongTimes", "correctTimes", "total_millisecond"]
        for c in need_num_cols:
            if c not in t.columns: t[c] = np.nan
        if "targetVocabulary" not in t.columns:
            t["targetVocabulary"] = pd.NA
        for c in need_num_cols:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        agg = t.groupby("user_sn").agg(
            word_isCorrect_mean=("isCorrect", lambda s: s.dropna().mean() if s is not None else 0.0),
            word_wrongTimes_mean=("wrongTimes", lambda s: s.dropna().mean() if s is not None else 0.0),
            word_correctTimes_mean=("correctTimes", lambda s: s.dropna().mean() if s is not None else 0.0),
            word_total_millisecond_mean=("total_millisecond", lambda s: s.dropna().mean() if s is not None else 0.0),
            word_vocab_unique_count=("targetVocabulary","nunique"),
        ).reset_index()
        word_feat = word_feat.merge(agg, on="user_sn", how="left")
    else:
        word_feat = pd.DataFrame({
            "user_sn": base["user_sn"],
            "word_isCorrect_mean": 0.0,
            "word_wrongTimes_mean": 0.0,
            "word_correctTimes_mean": 0.0,
            "word_total_millisecond_mean": 0.0,
            "word_vocab_unique_count": 0.0,
        })

    # H) dp003_math —— 欄位缺也能跑
    math_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp003_math.csv" in by_name:
        t = by_name["dp003_math.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        for col in ["is_correct","game_time"]:
            if col not in t.columns: t[col] = np.nan
            t[col] = pd.to_numeric(t[col], errors="coerce")
        agg = t.groupby("user_sn").agg(
            math_is_correct_mean=("is_correct", lambda s: s.dropna().mean() if s is not None else 0.0),
            math_game_time_mean_ms=("game_time", lambda s: s.dropna().mean() if s is not None else 0.0),
            math_unit_unique_count=("unit_id","nunique") if "unit_id" in t.columns else ("user_sn","size"),
            math_problem_count=("answer_problem_num","count") if "answer_problem_num" in t.columns else ("user_sn","count")
        ).reset_index()
        math_feat = math_feat.merge(agg, on="user_sn", how="left")
    else:
        for c in ["math_is_correct_mean","math_game_time_mean_ms","math_unit_unique_count","math_problem_count"]:
            math_feat[c] = 0.0

    # I) dp004_interaction
    inter_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp004_interaction.csv" in by_name:
        t = by_name["dp004_interaction.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        if "result_duration" in t.columns:
            t["duration_sec"] = t["result_duration"].map(parse_iso8601_duration)
        if "result_success" in t.columns:
            def to01(v):
                s = str(v).strip().lower()
                if s in ("true","1","y","yes"): return 1.0
                if s in ("false","0","n","no"): return 0.0
                return np.nan
            t["success01"] = t["result_success"].map(to01)
        agg = t.groupby("user_sn").agg(
            dp004_interaction_success_rate=("success01", lambda s: s.dropna().mean() if s is not None else 0.0) if "success01" in t.columns else ("user_sn","size"),
            dp004_interaction_duration_mean_sec=("duration_sec", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "duration_sec" in t.columns else ("user_sn","size"),
            dp004_interaction_count=("sn","count") if "sn" in t.columns else ("user_sn","count"),
            dp004_object_type_variety=("object_type","nunique") if "object_type" in t.columns else ("user_sn","size")
        ).reset_index()
        inter_feat = inter_feat.merge(agg, on="user_sn", how="left")
    for c in ["dp004_interaction_success_rate","dp004_interaction_duration_mean_sec","dp004_interaction_count","dp004_object_type_variety"]:
        if c not in inter_feat.columns: inter_feat[c] = 0.0

    # J) dp004_video
    video_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp004_video.csv" in by_name:
        t = by_name["dp004_video.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        if "result_duration" in t.columns:
            t["duration_sec"] = t["result_duration"].map(parse_iso8601_duration)
        counts = t.groupby(["user_sn","verb_name"]).size().unstack(fill_value=0) if "verb_name" in t.columns else None
        if counts is not None:
            for c in ["watched","paused"]:
                if c not in counts.columns: counts[c] = 0
            counts = counts.reset_index()
            counts["dp004_video_event_count"] = counts.drop(columns=["user_sn"]).sum(axis=1)
            tot = counts["dp004_video_event_count"].replace(0, 1.0)
            counts["dp004_video_pause_ratio"] = counts.get("paused", 0)/tot
        else:
            counts = pd.DataFrame({"user_sn": t["user_sn"].unique() if "user_sn" in t.columns else [],
                                   "dp004_video_event_count":0.0,"dp004_video_pause_ratio":0.0})
        agg = t.groupby("user_sn").agg(
            dp004_video_duration_mean_sec=("duration_sec", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean() if s is not None else 0.0) if "duration_sec" in t.columns else ("user_sn","size")
        ).reset_index()
        video_feat = video_feat.merge(agg, on="user_sn", how="left").merge(counts[["user_sn","dp004_video_event_count","dp004_video_pause_ratio"]], on="user_sn", how="left")
    for c in ["dp004_video_duration_mean_sec","dp004_video_event_count","dp004_video_pause_ratio"]:
        if c not in video_feat.columns: video_feat[c] = 0.0

    # K) dp004_webpage
    web_feat = pd.DataFrame({"user_sn": base["user_sn"]})
    if "dp004_webpage.csv" in by_name:
        t = by_name["dp004_webpage.csv"].copy()
        if "user_sn" in t.columns: t["user_sn"] = norm_id(t["user_sn"])
        agg = t.groupby("user_sn").agg(
            webpage_view_count=("sn","count") if "sn" in t.columns else ("user_sn","count"),
            webpage_resource_unique_count=("object_name","nunique") if "object_name" in t.columns else ("user_sn","size"),
            webpage_type_unique_count=("object_type","nunique") if "object_type" in t.columns else ("user_sn","size")
        ).reset_index()
        web_feat = web_feat.merge(agg, on="user_sn", how="left")
    for c in ["webpage_view_count","webpage_resource_unique_count","webpage_type_unique_count"]:
        if c not in web_feat.columns: web_feat[c] = 0.0

    # 合併所有特徵（以 dp001 活躍者基底為準）
    df = base[["user_sn"]].drop_duplicates()
    for part in [user_feat, review_feat, plus_feat, prac_feat, exam1_feat, exam2_feat,
                 word_feat, math_feat, inter_feat, video_feat, web_feat]:
        df = df.merge(part, on="user_sn", how="left")
    df = df.fillna(0.0)

    # 平均正確率（若有）
    acc_cols = [c for c in ["exam_accuracy","dp002_success_rate","word_isCorrect_mean","math_is_correct_mean","dp004_interaction_success_rate"] if c in df.columns]
    df["avg_correct_rate"] = df[acc_cols].mean(axis=1) if acc_cols else 0.0

    print(f"[DONE] 特徵工程完成：使用者 {len(df)} 人、特徵 {df.shape[1]-1} 欄（不含 user_sn）\n")
    return df

def onlypicture_kmeans_then_xgb(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    print("[INFO] 開始分群…")
    feature_cols = [c for c in df.columns if c!="user_sn" and pd.api.types.is_numeric_dtype(df[c])]
    assert feature_cols, "沒有可用的數值特徵可供分群。"
    var = df[feature_cols].var()
    keep = var[var > 1e-8].index.tolist()
    zero_rate = (df[keep] == 0).mean()
    keep = [c for c in keep if zero_rate[c] < 0.98]
    if len(keep) < len(feature_cols):
        print(f"  [INFO] 低變異/高零值欄移除：{len(feature_cols)-len(keep)} 欄")
    feature_cols = keep

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50, max_iter=500)
    km_labels = km.fit_predict(X)
    sil = silhouette_score(X, km_labels) if n_clusters>1 else np.nan
    print(f"  inertia: {km.inertia_:.2f} | silhouette: {sil:.3f}")

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.0,
        random_state=random_state, eval_metric="mlogloss",
        n_jobs=-1
    )
    xgb.fit(X, km_labels)
    xgb_pred = xgb.predict(X)
    cons = (xgb_pred==km_labels).mean()
    print(f"  XGB 與 KMeans 一致性: {cons:.2%}")

    out = df.copy()
    out["km_cluster"]  = km_labels
    out["xgb_cluster"] = xgb_pred
    name_map = {0:"Advanced",1:"Diligent",2:"Indifferent",3:"Persistent"}
    out["xgb_cluster_name"] = out["xgb_cluster"].map(lambda x: name_map.get(int(x), f"Cluster_{int(x)}"))

    vc = out["xgb_cluster_name"].value_counts()
    print("  群組規模：", vc.to_dict())
    print(f"  檢查總數：{vc.sum()} / rows = {len(out)}（應相等）")

    # 可選：輸出重要度
    try:
        import seaborn as sns
        importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp_path = BASE_DIR / "xgb_feature_importance.csv"
        importances.to_csv(imp_path, encoding="utf-8-sig")
        plt.figure(figsize=(10, max(4, min(20, len(importances)*0.25))))
        top_n = min(40, len(importances))
        sns.barplot(x=importances.values[:top_n], y=importances.index[:top_n])
        plt.title(f"XGBoost Feature Importance (top {top_n})")
        plt.xlabel("Importance"); plt.ylabel("Feature"); plt.tight_layout()
        plt.savefig(BASE_DIR / "xgb_feature_importance.png", dpi=180); plt.close()
    except Exception as e:
        print(f"[WARN] 畫特徵重要度失敗（不影響分群）：{e}")
    return out, feature_cols

def load_clusters_or_run_onlypicture(base: Path, n_clusters: int = 4) -> pd.DataFrame:
    cpath = base / "clusters_km_xgb.csv"
    if cpath.exists():
        df = read_csv_robust(cpath)
        need_cols = [c for c in ["user_sn","xgb_cluster_name","xgb_cluster"] if c in df.columns]
        assert "user_sn" in need_cols, "clusters_km_xgb.csv 缺 user_sn 欄位"
        out = df[need_cols].copy()
        if "xgb_cluster_name" not in out.columns:
            out["xgb_cluster_name"] = out["xgb_cluster"].map({0:"Advanced",1:"Diligent",2:"Indifferent",3:"Persistent"}).fillna(out["xgb_cluster"].astype(str))
        out["user_sn"] = norm_id(out["user_sn"])
        print(f"[INFO] 讀到分群：{len(out)} 位使用者")
        return out[["user_sn","xgb_cluster_name"]]

    print("[INFO] 未找到 clusters_km_xgb.csv，啟動 onlypicture 分群流程（dp001 基底）…")
    by_name = onlypicture_load_csvs(str(base))
    if not by_name:
        raise RuntimeError("onlypicture: 沒讀到任何檔案，無法分群。請確認 BASE_DIR。")
    feat = onlypicture_build_static_features(by_name)
    out_df, feature_cols = onlypicture_kmeans_then_xgb(feat, n_clusters=n_clusters, random_state=42)
    out_path = base / "clusters_km_xgb.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已輸出分群檔：{out_path}")
    out_df["user_sn"] = norm_id(out_df["user_sn"])
    return out_df[["user_sn","xgb_cluster_name"]]

# =========================
# 🔼🔼 onlypicture 分群（內嵌）到此為止 🔼🔼
# =========================

# --------------- 依分群重採樣 ---------------
def resample_by_given_groups(events: pd.DataFrame, clusters: pd.DataFrame, label_col: str = "xgb_cluster_name",
                             rule: str = "D", agg: str = "sum") -> dict[str, pd.Series]:
    m = events.merge(clusters[["user_sn", label_col]].rename(columns={label_col:"group"}), on="user_sn", how="inner")
    if m.empty:
        raise RuntimeError("事件表與分群對不上（merge 為空）。請檢查 user_sn 清洗。")
    m = m.set_index("ts")
    if agg == "mean":
        user_series = m.groupby(["group","user_sn"]).resample(rule)["weight"].sum().rename("signal").reset_index()
        grp = user_series.groupby(["group","ts"])["signal"].mean().rename("signal").reset_index()
    else:
        grp = m.groupby(["group"]).resample(rule)["weight"].sum().rename("signal").reset_index()
    out = {}
    for g, gdf in grp.groupby("group"):
        if gdf.empty: continue
        full_range = pd.date_range(gdf["ts"].min(), gdf["ts"].max(), freq=rule)
        s = gdf.set_index("ts")["signal"].reindex(full_range).fillna(0.0)
        s.name = g; out[g] = s
    return out

# --------------- 畫單一群的 CWT 圖 ---------------
def plot_one_group_wavelet(series: pd.Series, rule="D", save_dir: Path|None=None):
    if rule.upper().startswith("H"):
        dt = 1.0/24.0; unit="days"
    else:
        dt = 1.0; unit="days"
    sig = series.values.astype(float)
    sig = sig - sig.mean()

    if USE_PYWT:
        coeffs, freqs = pywt.cwt(sig, SCALES, 'morl', sampling_period=dt)
        power = (np.abs(coeffs))**2
        periods = 1.0 / freqs
    else:
        w = 6.0
        widths = (w * PERIODS_FOR_SCIPY) / (2*np.pi*dt)
        coeffs = cwt(sig, lambda M, s: morlet2(M, s, w=w), widths)
        power = (np.abs(coeffs))**2
        periods = PERIODS_FOR_SCIPY

    plt.figure(figsize=(8,4.2))
    im = plt.imshow(power,
                    extent=[0, len(sig), periods.min(), periods.max()],
                    aspect='auto', origin='lower',
                    vmax=np.percentile(power,99), vmin=np.percentile(power,1),
                    cmap='jet')
    plt.yscale('log')
    plt.title(f"Scalogram - {series.name}")
    plt.xlabel("Time Index")
    plt.ylabel(f"Period ({unit})")
    plt.colorbar(im, label="Power")
    plt.tight_layout()

    if save_dir:
        fn = f"wavelet_{series.name}.png".replace(" ","_")
        out = save_dir / fn
        plt.savefig(out, dpi=180)
        print(f"[OK] 輸出：{out}")
    plt.show(); plt.close()

# --------------- 主程式 ---------------
def main():
    assert BASE_DIR.exists(), f"資料夾不存在：{BASE_DIR}"

    clusters = load_clusters_or_run_onlypicture(BASE_DIR, n_clusters=4)
    print(f"[INFO] 群名預覽：{clusters['xgb_cluster_name'].value_counts(dropna=False).to_dict()}")

    events = build_events_table(BASE_DIR)
    if events.empty:
        raise RuntimeError("事件表為空：請先確認 dp001/2/3/4 原始檔是否可讀、是否有時間欄位。")

    series_dict = resample_by_given_groups(events, clusters, label_col=LABEL_COL, rule=RESAMPLE_RULE, agg=GROUP_AGG)
    if not series_dict:
        raise RuntimeError("重採樣後為空：請檢查分群群名與 user_sn 是否對得上。")

    save_dir = BASE_DIR
    group_order = [g for g in EXPECTED_GROUPS if g in series_dict] + [g for g in series_dict if g not in EXPECTED_GROUPS]
    for g in group_order:
        plot_one_group_wavelet(series_dict[g], rule=RESAMPLE_RULE, save_dir=save_dir)

    print("\n[DONE] 已完成四群小波圖輸出。")

if __name__ == "__main__":
    main()
