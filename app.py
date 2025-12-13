import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Inisialisasi Aplikasi ---
app = FastAPI(title="Customer Offer Prediction API", version="1.0")

# --- 1. Muat Model & Komponen (Load saat aplikasi start) ---
try:
    model = joblib.load('random_forest_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("✅ Semua model berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    # Penting: Pastikan file .joblib ada di folder yang sama dengan main.py

# --- 2. Definisikan Format Input (Schema) ---
# Ini memastikan data yang dikirim user sesuai format
class CustomerData(BaseModel):
    plan_type: str
    device_brand: str
    avg_data_usage_gb: float
    pct_video_usage: float
    avg_call_duration: float
    sms_freq: int
    monthly_spend: float  # Penting untuk feature engineering
    topup_freq: int
    travel_score: float
    complaint_count: int

    class Config:
        json_schema_extra = {
            "example": {
                 "plan_type": "Postpaid", 
                "device_brand": "Samsung",
                "avg_data_usage_gb": 8.0,    
                "pct_video_usage": 0.4,     
                "avg_call_duration": 2.0,    
                "sms_freq": 2,
                "monthly_spend": 75000,     
                "topup_freq": 1,
                "travel_score": 0.0,
                "complaint_count": 0
            }
        }

# --- 3. Endpoint Prediksi ---
@app.post("/predict")
def predict_offer(data: CustomerData):
    try:
        # A. Konversi ke DataFrame
        input_df = pd.DataFrame([data.dict()])

        # B. Feature Engineering
        if input_df['avg_data_usage_gb'].iloc[0] == 0:
            input_df['spend_per_gb'] = 0
        else:
            input_df['spend_per_gb'] = input_df['monthly_spend'] / input_df['avg_data_usage_gb']

        # C. Preprocessing
        processed_data = preprocessor.transform(input_df)

        # --- BAGIAN INI YANG BERUBAH (UPDATE) ---

        # 1. Dapatkan Label Prediksi (Juara 1)
        prediction_index = model.predict(processed_data)[0]
        predicted_label = label_encoder.inverse_transform([prediction_index])[0]

        # 2. Dapatkan Probabilitas (Keyakinan) -- BARU
        # predict_proba mengembalikan array, misal: [[0.1, 0.8, 0.1]]
        all_probs = model.predict_proba(processed_data)[0] 
        
        # Ambil nilai probabilitas tertinggi (Max value)
        confidence_score = float(np.max(all_probs)) 

        # Opsional: Ubah jadi format persen agar cantik (0.85 -> 85.0)
        confidence_percent = round(confidence_score * 100, 2)

        # ----------------------------------------

        return {
            "status": "success",
            "predicted_offer": predicted_label,
            "prediction_code": int(prediction_index),
            "confidence_score": confidence_score,     # Misal: 0.85
            "confidence_percent": f"{confidence_percent}%" # Misal: "85.0%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint Cek Kesehatan Server ---
@app.get("/")
def root():
    return {"message": "API Prediksi Penawaran Aktif"}