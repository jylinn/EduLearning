"""以本機讀取你自己的分群 / 生成建議 CSV；不需重新訓練模型。"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

SUPPORTED_GROUPS = {"Advanced", "Diligent", "Indifferent", "Persistent"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"user_sn": "string"})


def load_students(path: str | Path) -> pd.DataFrame:
    """相容原分析的 clusters_km_xgb.csv / 舊前端的 student_clusters_xgb.csv。"""
    frame = _read_csv(path)
    if "user_sn" not in frame:
        raise ValueError("找不到 user_sn 欄位，無法識別本機學生代號。")
    group_col = next((c for c in ["xgb_cluster_name", "label_xgb"] if c in frame.columns), None)
    if group_col is None:
        raise ValueError("找不到 xgb_cluster_name 或 label_xgb 欄位，請輸出分群表後再載入。")
    if group_col != "xgb_cluster_name":
        frame = frame.rename(columns={group_col: "xgb_cluster_name"})
    frame["user_sn"] = frame["user_sn"].str.strip().str.replace(r"\.0$", "", regex=True)
    frame = frame.dropna(subset=["user_sn", "xgb_cluster_name"]).copy()
    frame = frame[frame["user_sn"] != ""]
    frame["xgb_cluster_name"] = frame["xgb_cluster_name"].astype(str).str.strip()
    if frame.duplicated("user_sn").any():
        raise ValueError("CSV 中有重複 user_sn；請先釐清每位學生的分群紀錄。")
    if frame.empty:
        raise ValueError("檔案沒有可使用的學生紀錄。")
    return frame


def load_suggestions(path: str | Path) -> pd.DataFrame:
    """舊前端學生建議可供教師本機比對，但不會當成 RAG 知識證據。"""
    frame = _read_csv(path)
    if "user_sn" not in frame:
        raise ValueError("建議 CSV 需要 user_sn 欄位。")
    frame["user_sn"] = frame["user_sn"].str.strip().str.replace(r"\.0$", "", regex=True)
    return frame.drop_duplicates("user_sn")


def summarize_groups(frame: pd.DataFrame) -> str:
    items = frame["xgb_cluster_name"].value_counts().items()
    return "、".join(f"{name}: {number} 人" for name, number in items)


def fetch_student(frame: pd.DataFrame, user_sn: str | None) -> dict | None:
    if frame is None or frame.empty or not user_sn:
        return None
    m = frame[frame["user_sn"].astype(str) == str(user_sn)]
    return m.iloc[0].to_dict() if not m.empty else None
