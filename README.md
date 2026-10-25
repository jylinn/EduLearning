# 教育大數據｜教師個人化學習改善建議 RAG

> 基於使用者保留的兩個原始檔，**重新整合的新原型**，不是原組員 RAG 程式的復原，也未實際接觸學生原始資料。

## 為何重建？

老師選學生、詢問「這位學生可以怎麼改善學習？」→ 根據現有的學生**分析結果**產生匿名行為摘要 → 從本機教學策略資料庫檢索相關文字段落 → Gemini 根據檢索結果撰寫建議，顯示引用來源。 這才有 RAG 的「Retrieve + Generate」。

原本的 `topic_back (3).py` 只有把分群與特徵直接交給 Gemini 生成建議，**沒有知識庫檢索**。它保存在 `legacy/topic_back_original.py`，未強行修改原作者程式。原本的特徵工程、KMeans、XGBoost、Wavelet 分析程式也原封保存在 `analysis/2025edu_data.py`。

## Windows 11 CMD 快速啟動

```cmd
cd /d "你解壓縮後的資料夾"
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
```

瀏覽器會開啟 `http://127.0.0.1:7860`。預設僅監聽 `127.0.0.1`，不對公網開放。

## 操作方式

1. 在「① 分析結果」匯入你原分析程式的 `clusters_km_xgb.csv`，或舊前端輸出的 `student_clusters_xgb.csv`；後者只有類別時無法給出特徵層級的個別分析。
2. 選擇學生代號，確認讀取到的分析特徵。
3. 在「② 教師詢問改善建議」輸入教學問題，按「查詢知識庫並產生改善建議」。
4. 沒有 API 金鑰時只會顯示**檢索預覽**，不是完整 RAG 生成。
5. 若要比對舊版直接 LLM 產生的建議，另在①匯入 `student_suggestions.csv`。舊建議不當作知識庫的可靠證據。

## 使用 Gemini 生成

先取得 Gemini API Key。在 Windows CMD 的**當次終端機**輸入（不要寫入 Git 追蹤的檔案）：

```cmd
set GEMINI_API_KEY=你的金鑰
set GEMINI_MODEL=gemini-3.8-flash
python app.py
```

以上模型名稱依 2026 年 Google 官方文件示例；不同帳號的模型可用性可能不同。可改 `GEMINI_MODEL`。API 是 `google-genai` 官方 SDK `client.models.generate_content`。

**隱私限制：**這個原型只會自動把非識別性的部分行為指標傳至 Gemini，不會在程式中加入 `user_sn`、原始三科分數、原始學習日誌；若教師在問句自己輸入學生真名等個資，原型未必能完全辨識，請勿輸入。教育資料對外傳送需先確認校方與資料授權、倫理審查及適用法規。要正式用於真實學生，須加入登入認證、教師存取控制、稽核、匿名策略和可信來源評估。

## 專案檔案

```text
app.py                         新 Gradio 教師問答前端
analysis_adapter.py            支援新舊兩種分群結果 CSV
rag.py                         本機中文 TF-IDF 檢索 + Gemini 建議生成
knowledge/*.md                 僅為功能測試的示範策略，非核對文獻
analysis/2025edu_data.py       你保留的完整分析主程式（未改）
legacy/topic_back_original.py  你剛補上的 Gradio + Gemini 原版程式（未改）
data/demo_clusters.csv         虛構示範數據
tests/test_teacher_rag.py      本機自動測試
```

## 模型及結果注意事項

- 原程式把 KMeans 群號硬對應 Advanced / Diligent / Indifferent / Persistent，但 KMeans 數字不具固定含義，應以群組中心與特徵解釋再命名。
- XGBoost 在 KMeans 訓練標籤上訓練並在同一批資料預測，不能以一致性當成泛化準確率。
- 舊版前端將不同 CSV 直接縱向串接，以及將 timestamp 直接 diff，可能與實際多表欄位格式不合；本版**沒有宣稱**這些原始流程已經修復。這個 app 以既有分析輸出為接口，避免改動舊演算法。
- RAG 知識庫目前是**示範文字**。做論文或作品集時，需補上可核對的文獻、授權、引用欄位、檢索品質評估與回答正確性評估。
- 研究結果與教材若涉及學校資料授權，**不要直接提交到 Public GitHub**；`data/outputs/` 與 `private_data/` 被排除。

## 本地檢驗範圍

測試涵蓋 Python 語法、分群 CSV 格式相容、檢索功能、部分匿名化、無 API 金鑰時的行為；不代表已與真實 CSV、原版分析的所有依賴、或 Gemini 線上回覆做端到端驗證。
