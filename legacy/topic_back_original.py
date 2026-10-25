import os
import re
import json
import time
import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 載入 API Key
load_dotenv()
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")

# 包含重試與較長 timeout 的 Gemini 呼叫函式
def call_gemini_api(api_key, prompt_text, retries=2, backoff=1):
    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-1.5-pro:generateContent?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            if text.strip().startswith("```json"):
                text = re.sub(r"```json|```", "", text).strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise ValueError("❌ 找不到合法 JSON 格式回應")
            return json.loads(m.group(0))
        except Exception as e:
            print(f"[Gemini Attempt {attempt}] Error: {e}")
            last_exc = e
            if attempt < retries:
                time.sleep(backoff)
    raise last_exc


def analyze_student_data(files, api_key):
    # 1. 讀取 CSV 並統一 user_sn
    dfs = []
    for f in files:
        df_ = pd.read_csv(f.name)
        for c in df_.columns:
            if c.strip().lower() == "user_sn":
                df_.rename(columns={c: "user_sn"}, inplace=True)
        dfs.append(df_)
    master = pd.concat(dfs, ignore_index=True)

    # 2. 計算點擊與時間比例行為特徵
    behavior_cols = [
        "SkippingBackward_freq",
        "SkippingForward_freq",
        "Pause_freq",
        "SkippingBackward_time_ratio",
        "SkippingForward_time_ratio",
        "Pause_time_ratio"
    ]
    review_df = next((d for d in dfs if {"user_sn","review_sn"}.issubset(d.columns)), None)
    plus_df   = next((d for d in dfs if {"review_sn","view_action","timestamp"}.issubset(d.columns)), None)
    if review_df is not None and plus_df is not None:
        merged = plus_df.merge(
            review_df[["review_sn","user_sn"]], on="review_sn", how="left"
        ).dropna(subset=["user_sn"])
        # 次數
        counts = merged.groupby(["user_sn","view_action"]).size().unstack(fill_value=0)
        for act in ["dragleft","dragright","paused"]:
            if act not in counts.columns:
                counts[act] = 0
        cf = counts[["dragleft","dragright","paused"]].reset_index()
        cf.columns = ["user_sn"] + behavior_cols[:3]
        # 時間比例
        merged = merged.sort_values(["user_sn","timestamp"])
        merged['duration'] = merged.groupby('user_sn')['timestamp'].diff().fillna(0)
        total_time = merged.groupby('user_sn')['duration'].sum().replace(0,1)
        tr = merged.groupby(['user_sn','view_action'])['duration'].sum().unstack(fill_value=0)
        for act, col in zip(["dragleft","dragright","paused"], behavior_cols[3:]):
            cf[col] = tr.get(act, 0) / total_time
    else:
        cf = pd.DataFrame({"user_sn": master["user_sn"].unique()})
        for col in behavior_cols:
            cf[col] = 0.0

    # 3. 計算 turbo 特徵
    turbo_cols = [
        "turbo_event_count","turbo_mean_speed","turbo_std_speed","turbo_unique_speeds",
        "turbo_0.5_count","turbo_1.0_count","turbo_1.5_count",
        "turbo_0.5_ratio","turbo_1.0_ratio","turbo_1.5_ratio"
    ]
    if review_df is not None and plus_df is not None and "turbo" in plus_df.columns:
        tf = plus_df.merge(
            review_df[["review_sn","user_sn"]], on="review_sn", how="left"
        ).dropna(subset=["user_sn","turbo"])
        tf["turbo"] = tf["turbo"].astype(float)
        grp = tf.groupby("user_sn")
        turbo_feat = grp["turbo"].agg(
            turbo_event_count="count",
            turbo_mean_speed="mean",
            turbo_std_speed="std",
            turbo_unique_speeds=lambda s: s.nunique()
        ).fillna(0).reset_index()
        speeds = [0.5,1.0,1.5]
        for sp in speeds:
            cnt = tf[tf["turbo"]==sp].groupby("user_sn").size()
            turbo_feat[f"turbo_{sp}_count"] = cnt.fillna(0)
        tot = turbo_feat["turbo_event_count"].replace(0,1)
        for sp in speeds:
            turbo_feat[f"turbo_{sp}_ratio"] = turbo_feat[f"turbo_{sp}_count"] / tot
    else:
        turbo_feat = pd.DataFrame({"user_sn": master["user_sn"].unique()})
        for col in turbo_cols:
            turbo_feat[col] = 0.0

    # 4. 學業三科平均
    score_cols = ["chinese_score","math_score","english_score"]
    present_scores = [c for c in score_cols if c in master.columns]
    if present_scores:
        df_scores = master.groupby("user_sn", as_index=False)[present_scores].mean()
    else:
        df_scores = pd.DataFrame({"user_sn": master["user_sn"].unique()})
        for c in score_cols:
            df_scores[c] = 0.0

    # 5. 合併並標準化
    df = df_scores.merge(cf, on="user_sn").merge(turbo_feat, on="user_sn").fillna(0)
    features = behavior_cols + present_scores + turbo_cols
    X = StandardScaler().fit_transform(df[features])

    # 6. 分群
    df["cluster_km"]  = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1024).fit_predict(X)
    xgb = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb.fit(X, df["cluster_km"])
    df["cluster_xgb"] = xgb.predict(X)
    name_map = {0:"Advanced",1:"Diligent",2:"Indifferent",3:"Persistent"}
    df["label_xgb"] = df["cluster_xgb"].map(name_map)

    # 7. 建議
    prompts, suggestions = [], []
    for _, r in df.iterrows():
        snippet = {k: float(r[k]) for k in features}
        s = json.dumps(snippet, ensure_ascii=False)
        s = (s[:100]+"...") if len(s)>100 else s
        prompts.append(
            f"""
你是一位教育顧問。
學生 SN：\"{r['user_sn']}\"，群組：\"{r['label_xgb']}\"。
指標：{s}

請回純 JSON：{{\"advantage\":\"優點\",\"weakness\":\"待改進\",\"suggestion\":\"建議\"}}
"""
        )
    with ThreadPoolExecutor(max_workers=1) as exe:
        fut2idx = {exe.submit(call_gemini_api, api_key, p): i for i,p in enumerate(prompts)}
        for _ in range(len(prompts)):
            suggestions.append(None)
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                suggestions[i] = fut.result()
            except:
                suggestions[i] = {"advantage":"無法生成","weakness":"無法生成","suggestion":"無法生成"}
    sug_df = pd.DataFrame(suggestions)
    sug_df.insert(0,"user_sn", df["user_sn"])
    sug_df.insert(1,"cluster", df["label_xgb"])

    # 8. 四群長條圖：只點擊頻次 + 各科平均
    bar_files = []
    plot_keys = behavior_cols[:3] + present_scores
    for lab in name_map.values():
        sub = df[df['label_xgb']==lab]
        vals = [sub[k].mean() if not sub.empty else 0 for k in plot_keys]
        plt.figure(figsize=(6,4))
        sns.barplot(x=plot_keys, y=vals)
        plt.title(f"{lab} Clicks & Scores")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fn=f"bar_{lab}.png"
        plt.savefig(fn)
        plt.close()
        bar_files.append(fn)

    # 9. 輸出
    df[["user_sn","label_xgb"]].to_csv("student_clusters_xgb.csv",index=False,encoding="utf-8-sig")
    sug_df.to_csv("student_suggestions.csv",index=False,encoding="utf-8-sig")

    return (
        "student_clusters_xgb.csv","student_suggestions.csv",
        bar_files[0],bar_files[1],bar_files[2],bar_files[3],
        sug_df[["user_sn","cluster","advantage","weakness","suggestion"]]
    )


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 學生個人化學習分析與建議")
        files   = gr.File(file_types=[".csv"], file_count="multiple", label="📂 上傳 CSV")
        api_key = gr.Textbox(value=DEFAULT_API_KEY, type="password", label="🔑 API Key")
        btn     = gr.Button("開始分析")

        out1 = gr.File(label="分群結果 CSV")
        out2 = gr.File(label="建議結果 CSV")
        out3 = gr.Image(label="Advanced Clicks & Scores")
        out4 = gr.Image(label="Diligent Clicks & Scores")
        out5 = gr.Image(label="Indifferent Clicks & Scores")
        out6 = gr.Image(label="Persistent Clicks & Scores")
        table = gr.Dataframe(label="建議摘要表格")

        btn.click(
            analyze_student_data,
            inputs=[files, api_key],
            outputs=[out1,out2,out3,out4,out5,out6,table]
        )
    return demo

if __name__=="__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
