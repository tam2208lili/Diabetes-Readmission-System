# 🏥 Clinical Decision Support System: Diabetes Readmission AI

### 🚀 Project Overview
This project delivers an end-to-end Machine Learning solution to predict the 30-day readmission risk for diabetic patients. The system is designed not just as a predictive model, but as a strategic tool for hospital resource management and patient safety enhancement.

### 📊 Dataset Information
The model is trained on **ten years of patient information** with the following features:
- **Source:** [Kaggle - Hospital Readmission Dataset](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)
- **Demographics & Stay:** `age` (bracket), `time_in_hospital` (1-14 days).
- **Clinical Counts:** `n_procedures`, `n_lab_procedures`, `n_medications`.
- **History:** `n_outpatient`, `n_inpatient`, `n_emergency` visits in the year before the stay.
- **Diagnosis:** `medical_specialty` (admitting physician), `diag_1` (primary), `diag_2`, `diag_3`.
- **Tests & Treatment:** `glucose_test` (serum level), `A1Ctest` (A1C level), `change` in diabetes medication, and `diabetes_med` (prescription status).
- **Target:** `readmitted` (status of being readmitted to the hospital).


### 🧠 Strategic Modeling: "Patient-First" Philosophy
In healthcare, the cost of a **False Negative** (missing a high-risk patient) is significantly higher than a **False Positive** (monitoring a healthy patient). 

**Key Technical Decisions:**
- **Recall Optimization (83%):** I intentionally tuned the model using **XGBoost** and adjusted the decision threshold to **0.4**. This ensures the system acts as a "safety net," catching nearly all potential readmissions.
- **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns effectively from rare readmission cases.
- **Feature Engineering:** Integrated clinical intensity metrics (e.g., interaction between inpatient stays and time in hospital) via SQL.

### 💰 Operational & Economic Considerations
A high-recall model introduces a trade-off in **Precision (52%)**. I have accounted for the following operational impacts:
1. **Resource Allocation:** While the model triggers more "alerts," it allows hospitals to prevent expensive emergency re-hospitalizations, which far outweigh the cost of a preventative follow-up call or a 15-minute nurse consultation.
2. **Efficiency:** By filtering out the bottom 50% of low-risk patients with high confidence, the hospital can focus its limited discharge-support staff where they are needed most.

### 🛠️ Tech Stack
- **Languages:** Python (Pandas, Scikit-learn, XGBoost)
- **Database:** SQL (SQLite for clinical data standardization)
- **Interface:** Streamlit (Interactive Clinical Dashboard)
- **Deployment:** GitHub & Streamlit Cloud

### 📈 Performance Metrics
- **ROC-AUC Score:** 0.644
- **Recall (Sensitivity):** 0.826
- **Brier Score:** 0.235 (Ensuring well-calibrated risk probabilities)
