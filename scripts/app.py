import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION ---
# Đường dẫn tuyệt đối khớp với cấu trúc trên GitHub Codespaces của bạn
BASE_DIR = "/workspaces/Diabetes-Readmission-System"
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Thiết lập trang giao diện rộng và chuyên nghiệp
st.set_page_config(
    page_title="Clinical Risk Support", 
    page_icon="🏥", 
    layout="wide"
)

# --- 2. LOAD ASSETS (Sử dụng Cache để tăng tốc độ) ---
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

model, encoders = load_assets()

# --- 3. UI HEADER ---
st.title("🏥 Patient Readmission Risk Predictor")
st.markdown("""
This AI-powered tool assists hospital staff in identifying diabetic patients with a high risk of 30-day readmission.
*Focus: **High-Sensitivity Screening (Recall-Optimized)***
""")
st.write("---")

if model is None:
    st.error("❌ Model files not found! Please run 'python scripts/model_training.py' first.")
    st.stop()

# --- 4. INPUT INTERFACE ---
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Patient Info")
        age = st.number_input("Patient Age", 0, 100, 65)
        specialty = st.selectbox("Medical Specialty", encoders['specialty'].classes_)
        diag = st.selectbox("Primary Diagnosis (Diag_1)", encoders['diag_1'].classes_)

    with col2:
        st.subheader("📋 Clinical History")
        time_hosp = st.slider("Time in Hospital (Days)", 1, 14, 3)
        n_inpatient = st.number_input("Prior Inpatient Visits (Last Year)", 0, 20, 0)
        n_emergency = st.number_input("Prior Emergency Visits (Last Year)", 0, 20, 0)

    with col3:
        st.subheader("🧪 Lab & Treatment")
        a1c = st.selectbox("A1C Test Result", encoders['A1Ctest'].classes_)
        glucose = st.selectbox("Glucose Test Result", encoders['glucose_test'].classes_)
        med_change = st.selectbox("Medication Change", encoders['change'].classes_)

# --- 5. PREDICTION LOGIC ---
st.write("---")
if st.button("🚀 Analyze Readmission Risk", use_container_width=True):
    # Tạo DataFrame từ input (phải khớp các cột với model_training)
    input_data = pd.DataFrame({
        'age_numeric': [age],
        'time_in_hospital': [time_hosp],
        'n_lab_procedures': [45],  # Giá trị trung bình mặc định
        'n_medications': [15],     # Giá trị trung bình mặc định
        'n_inpatient': [n_inpatient],
        'n_emergency': [n_emergency],
        'hosp_intensity': [n_inpatient * time_hosp], # Feature engineering từ SQL logic
        'specialty': [specialty],
        'diag_1': [diag],
        'glucose_test': [glucose],
        'A1Ctest': [a1c],
        'change': [med_change],
        'diabetes_med': ['yes']     # Mặc định yes cho bệnh nhân tiểu đường
    })

    # Encode dữ liệu chữ sang số dựa trên bộ từ điển đã lưu
    for col in encoders:
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])

    # Dự báo xác suất
    risk_proba = model.predict_proba(input_data)[0][1]

    # --- 6. DISPLAY RESULTS ---
    st.subheader("📊 Risk Assessment Result")
    
    # Hiển thị thanh tiến trình rủi ro
    st.progress(float(risk_proba))
    
    # Phân loại dựa trên ngưỡng 0.4 (Threshold đã tối ưu Recall)
    if risk_proba >= 0.4:
        st.error(f"### HIGH RISK: {risk_proba*100:.1f}%")
        st.markdown("""
        **Clinical Recommendation:**
        - Assign a dedicated case manager for discharge planning.
        - Schedule a follow-up call within 48 hours.
        - Review medication adherence with the patient.
        """)
    else:
        st.success(f"### LOW RISK: {risk_proba*100:.1f}%")
        st.markdown("**Clinical Recommendation:** Proceed with standard discharge protocol.")

# --- 7. FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for clinical decision support and should not replace professional medical judgment.")

# Run with this code in terminal
# streamlit run /workspaces/Diabetes-Readmission-System/scripts/app.py