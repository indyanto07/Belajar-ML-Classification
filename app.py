from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- LOAD MODEL & SCALER ---
try:
    joblib_model = joblib.load('SVM_model.joblib')
    print("✅ Model SVM berhasil dimuat!")
except Exception as e:
    print("❌ ERROR: Gagal memuat SVM_model.joblib", e)
    joblib_model = None

try:
    # Load Scaler (Pastikan file scaler.joblib sudah ada!)
    scaler = joblib.load('scaler.joblib')
    print("✅ Scaler berhasil dimuat!")
except Exception as e:
    print("⚠️ WARNING: Scaler tidak ditemukan! Prediksi mungkin tidak akurat.", e)
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    print("\n--- Menerima Request Baru ---")
    if not joblib_model:
        return jsonify({'error': 'Model belum dimuat di server'}), 500

    try:
        req_data = request.json
        if not req_data or 'data_nasabah_baru' not in req_data:
            return jsonify({'error': "Data JSON harus punya key 'data_nasabah_baru'"}), 400
        
        raw_data = req_data['data_nasabah_baru']
        df = pd.DataFrame(raw_data)
        
        # --- PREPROCESSING ---
        
        # 1. Drop Kolom
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # 2. Encoding
        if 'Geography' in df.columns:
            df['Geography'] = df['Geography'].astype(str).str.strip().replace({'France': 0, 'Germany': 1, 'Spain': 2})
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype(str).str.strip().replace({'Female': 0, 'Male': 1})
            
        # 3. Urutkan Kolom (Wajib sama dengan training)
        expected_order = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        df = df.reindex(columns=expected_order, fill_value=0)
        
        # 4. Convert ke Float
        X = df.astype(float)

        # 5. SCALING (PENTING!)
        if scaler:
            # Gunakan scaler yang sama dengan training untuk mengubah angka besar jadi kecil
            X_scaled = scaler.transform(X)
            print("Data berhasil di-scale.")
        else:
            X_scaled = X
            print("WARNING: Melakukan prediksi TANPA scaling.")

        # --- PREDIKSI ---
        prediction = joblib_model.predict(X_scaled)
        
        return jsonify({
            'status': 'success',
            'jumlah_data': len(prediction),
            'prediksi_churn': prediction.tolist()
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)