"""教師導向 Gradio Dashboard：讀取離線分析成果 + RAG 教師改善建議。"""
from __future__ import annotations

from pathlib import Path
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from analysis_adapter import load_students, load_suggestions, summarize_groups, fetch_student
from rag import answer_teacher, METRICS

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "data" / "demo_clusters.csv"
OUTPUTS = ROOT / "data" / "outputs"


def load_results(file: str | None):
    path = Path(file) if file else OUTPUTS / "clusters_km_xgb.csv"
    is_demo = False
    if not path.exists():
        path, is_demo = DEMO, True
    try:
        df = load_students(path)
        options = df["user_sn"].tolist()
        selected = options[0] if options else None
        hint = "**目前使用的是人工虛構示範資料。**" if is_demo else "**已讀取你提供的分析結果。**"
        status = f"{hint} 共 {len(df)} 筆。{summarize_groups(df)}。群名須經實際特徵驗證。"
        plot = group_plot(df)
        return df, gr.update(choices=options, value=selected), status, plot
    except Exception as exc:
        return pd.DataFrame(), gr.update(choices=[], value=None), f"載入失敗：{exc}", None


def group_plot(df: pd.DataFrame):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.3))
    order = df["xgb_cluster_name"].value_counts()
    ax.bar(order.index.tolist(), order.values.tolist())
    ax.set_title("Distribution of exploratory groups")
    ax.set_ylabel("Students")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def show_student(student_id: str | None, df: pd.DataFrame | None, sug: pd.DataFrame | None):
    row = fetch_student(df, student_id)
    if row is None:
        return "請選擇學生。", pd.DataFrame(columns=["指標", "值"]), ""
    metrics = [{"指標": label, "值": row[key]} for key, label in METRICS.items() if key in row]
    msg = f"**分群：{row['xgb_cluster_name']}**（探索性命名，非學習能力診斷）"
    if not metrics:
        msg += "\n\n這份分群 CSV 只有標籤，沒有學生的個別特徵。要做精準的建議，請改用主分析程式輸出的完整 `clusters_km_xgb.csv`。"
    prior = ""
    if sug is not None and not sug.empty:
        match = sug[sug["user_sn"] == str(student_id)]
        if not match.empty:
            r = match.iloc[0]
            prior = "\n\n".join(f"**{c}：** {r[c]}" for c in ["advantage", "weakness", "suggestion"] if c in r)
    return msg, pd.DataFrame(metrics, columns=["指標", "值"]), prior or "（尚未匯入舊前端產生的建議結果）"


def import_prior(filepath: str | None):
    if not filepath:
        return pd.DataFrame(), "未提供舊建議檔。"
    try:
        frame = load_suggestions(filepath)
        return frame, f"已載入 {len(frame)} 筆舊建議（僅供比對，不當作檢索知識來源）。"
    except Exception as exc:
        return pd.DataFrame(), f"讀取舊建議失敗：{exc}"


def ask_teacher(user_sn: str | None, q: str, df: pd.DataFrame | None):
    row = fetch_student(df, user_sn)
    if row is None:
        return "請先在『分析結果』頁籤載入 CSV 並選取學生。", "", ""
    return answer_teacher(q, row)


with gr.Blocks(title="教育大數據：教師個人化建議 RAG") as demo:
    gr.Markdown("# 教育大數據｜學生分群分析與教師 RAG 學習建議\n"
                "本機教學展示版本：保留你的分析演算法與舊 Gradio 原始碼；教師依分群與可用特徵詢問改善方式。")
    dataset = gr.State(pd.DataFrame())
    prior_suggestions = gr.State(pd.DataFrame())
    with gr.Tab("① 分析結果"):
        analysis_file = gr.File(label="匯入 clusters_km_xgb.csv（或 student_clusters_xgb.csv）", file_types=[".csv"], type="filepath")
        load_btn = gr.Button("載入分析成果", variant="primary")
        status = gr.Markdown("請上傳分析結果 CSV；直接按載入則使用人工示範資料。")
        dist = gr.Plot(label="四群分布")
        student = gr.Dropdown(label="選擇學生代號（本機顯示，不傳送給 LLM）", choices=[], interactive=True)
        group_desc = gr.Markdown()
        feature_table = gr.Dataframe(label="可供教師檢視的分析特徵（僅本機顯示）", interactive=False)
        prior_file = gr.File(label="（選用）匯入舊前端 student_suggestions.csv", file_types=[".csv"], type="filepath")
        prior_btn = gr.Button("匯入舊建議")
        prior_status = gr.Markdown()
        prior_text = gr.Markdown()
        gr.Markdown("如果要顯示 Wavelet 圖片，可從 `analysis/2025edu_data.py` 的輸出結果挑選並放進報告；此頁先聚焦教師問答。")
    with gr.Tab("② 教師詢問改善建議（RAG）"):
        gr.Markdown("老師先在①選學生，再問：『這位學生的學習行為可以怎麼改善？』"
                    "\n\n檢索文件目前是示範教學素材，不能宣稱有學術證據。若未設定 Gemini 金鑰，只會展示檢索結果。")
        question = gr.Textbox(label="老師的問題（請勿輸入真名或其他個資）", lines=3,
                              value="根據這位學生的學習行為，老師可以從哪些方面協助他改善學習？請提供具體做法與觀察方式。")
        ask_btn = gr.Button("查詢知識庫並產生改善建議", variant="primary")
        answer = gr.Markdown(label="教師建議")
        evidence = gr.Markdown(label="檢索來源")
        local_summary = gr.Textbox(label="本機特徵摘要（不直接傳送學生編號）", lines=5, interactive=False)
    with gr.Tab("③ 使用說明"):
        gr.Markdown("""### 舊前端與新 RAG 的差別
- 你提供的 `legacy/topic_back_original.py`：可上傳 CSV，做 MiniBatchKMeans、XGBoost，再直接請 Gemini 生成建議（原版**不是**真正 RAG）。
- `analysis/2025edu_data.py`：你現有的特徵工程、KMeans、XGBoost、Wavelet 主程式。
- 本程式 `app.py`：以已有模型輸出為基礎，補上教師查詢 + 知識庫檢索 + Gemini 生成及來源顯示。

### 使用原分析結果
先在你自己的機器執行主分析程式，產生 `clusters_km_xgb.csv`，再於①匯入。上傳真實教育資料時請只在獲授權的本機操作，**不要公開學生原始資料或部署無登入驗證的網頁**。

### 學術提醒
原本 0～3 群的命名只是程式對應，不是經過驗證的學習能力結論；教師必須參考實際特徵與自己的觀察。""")
    load_btn.click(load_results, inputs=[analysis_file], outputs=[dataset, student, status, dist])
    student.change(show_student, inputs=[student, dataset, prior_suggestions], outputs=[group_desc, feature_table, prior_text])
    prior_btn.click(import_prior, inputs=[prior_file], outputs=[prior_suggestions, prior_status]).then(
        show_student, inputs=[student, dataset, prior_suggestions], outputs=[group_desc, feature_table, prior_text])
    ask_btn.click(ask_teacher, inputs=[student, question, dataset], outputs=[answer, evidence, local_summary])

if __name__ == "__main__":
    # 避免無驗證的學生資料服務暴露到公網。
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
