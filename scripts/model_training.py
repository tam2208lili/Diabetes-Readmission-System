# Install libraries in Terminal
# pip install pandas scikit-learn matplotlib seaborn joblib streamlit imbalanced-learn xgboost

import pandas as pd
import sqlite3
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    classification_report, 
    recall_score, 
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- 1. CONFIGURATION & PATHS ---
# Sử dụng đường dẫn tuyệt đối cho GitHub Codespaces
BASE_DIR = "/workspaces/Diabetes-Readmission-System"
CSV_PATH = os.path.join(BASE_DIR, "data/hospital_readmissions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# --- 2. DATA EXTRACTION & SQL PRE-PROCESSING ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Could not find CSV at {CSV_PATH}")

df_raw = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(':memory:')
df_raw.to_sql('hospital_data', conn, index=False, if_exists='replace')

# Truy vấn SQL để xử lý đặc trưng (Feature Engineering)
query = """
SELECT 
    CASE 
        WHEN age = '[0-10)' THEN 5 WHEN age = '[10-20)' THEN 15 
        WHEN age = '[20-30)' THEN 25 WHEN age = '[30-40)' THEN 35
        WHEN age = '[40-50)' THEN 45 WHEN age = '[50-60)' THEN 55
        WHEN age = '[60-70)' THEN 65 WHEN age = '[70-80)' THEN 75
        WHEN age = '[80-90)' THEN 85 WHEN age = '[90-100)' THEN 95
    END AS age_numeric,
    time_in_hospital, n_lab_procedures, n_medications, n_inpatient, n_emergency,
    -- Tạo biến tương tác (Interaction Term) để tăng Recall
    (n_inpatient * time_in_hospital) as hosp_intensity,
    CASE WHEN medical_specialty = 'Missing' THEN 'Unknown' ELSE medical_specialty END AS specialty,
    diag_1, glucose_test, A1Ctest, change, diabetes_med,
    CASE WHEN readmitted = 'yes' THEN 1 ELSE 0 END AS target
FROM hospital_data
"""
df = pd.read_sql(query, conn)

# --- 3. CATEGORICAL ENCODING ---
le_dict = {}
cat_cols = ['specialty', 'diag_1', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# --- 4. DATA SPLITTING ---
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. HANDLING IMBALANCE WITH SMOTE ---
# Tạo thêm mẫu dữ liệu nhân tạo cho nhóm thiểu số (nhóm tái nhập viện)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- 6. MODEL TRAINING (XGBOOST) ---
# Sử dụng XGBoost - Thuật toán mạnh mẽ nhất cho dữ liệu bảng
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# --- 7. ADVANCED EVALUATION ---
# Tính toán xác suất
probs = model.predict_proba(X_test)[:, 1]

# Sử dụng ngưỡng (Threshold) 0.4 để ưu tiên Recall (Độ nhạy)
custom_threshold = 0.4
preds_custom = (probs >= custom_threshold).astype(int)

print("\n" + "="*40)
print("📊 FINAL MODEL PERFORMANCE REPORT")
print("="*40)

# Các chỉ số cốt lõi
print(f"1. ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
print(f"2. Brier Score:   {brier_score_loss(y_test, probs):.4f}")
print(f"3. Recall Score:  {recall_score(y_test, preds_custom):.4f}")

# Báo cáo chi tiết
print("\n📋 Detailed Classification Report (Threshold = 0.4):")
print(classification_report(y_test, preds_custom))

# Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, preds_custom)
print("🧩 Confusion Matrix:")
print(f"   [True Negatives  : {cm[0][0]}] | [False Positives : {cm[0][1]}]")
print(f"   [False Negatives : {cm[1][0]}] | [True Positives  : {cm[1][1]}]")

print("="*40)

# --- 8. EXPORT MODEL COMPONENTS ---
joblib.dump(model, MODEL_PATH)
joblib.dump(le_dict, ENCODER_PATH)
print(f"\n✅ Model and Encoders successfully saved to: {BASE_DIR}")