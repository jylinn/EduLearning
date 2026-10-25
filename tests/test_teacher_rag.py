import sys
import unittest
from pathlib import Path
import tempfile
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_adapter import load_students, fetch_student
from rag import retrieve, answer_teacher, format_prompt, metric_summary


class TeacherRagTests(unittest.TestCase):
    def test_legacy_frontend_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "student_clusters_xgb.csv"
            pd.DataFrame({"user_sn": ["00007"], "label_xgb": ["Diligent"]}).to_csv(p, index=False)
            profile = fetch_student(load_students(p), "00007")
            self.assertEqual(profile["xgb_cluster_name"], "Diligent")

    def test_original_analysis_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clusters_km_xgb.csv"
            pd.DataFrame({"user_sn": ["9001"], "xgb_cluster_name": ["Persistent"], "Pause_freq": [12]}).to_csv(p, index=False)
            profile = fetch_student(load_students(p), "9001")
            self.assertEqual(profile["Pause_freq"], 12)

    def test_retrieval(self):
        self.assertTrue(retrieve("暫停影片如何改善學習安排？", {"Pause_freq": 12}))

    def test_private_prompt(self):
        profile = {"user_sn": "9001", "xgb_cluster_name": "Advanced", "Pause_freq": 12,
                   "english_score": 51, "chinese_score": 72}
        hits = retrieve("學生 9001 暫停影片怎麼改善？", profile)
        prompt = format_prompt("學生 9001 暫停影片怎麼改善？", profile, hits)
        self.assertNotIn("9001", prompt)
        self.assertNotIn("english_score", prompt)
        self.assertNotIn("51", metric_summary(profile, safe_for_llm=True))

    def test_keyless_mode(self):
        import os
        prior = os.environ.pop("GEMINI_API_KEY", None)
        try:
            answer, sources, _ = answer_teacher("老師如何協助經常暫停影片的學生？", {"Pause_freq": 15})
            self.assertIn("檢索預覽", answer)
            self.assertIn("來源", sources)
        finally:
            if prior is not None:
                os.environ["GEMINI_API_KEY"] = prior


if __name__ == "__main__":
    unittest.main()
