import requests
import json
import os

url = "http://127.0.0.1:5000/predict"

# 1. Buka file JSON yang sudah kamu buat sebelumnya
filename = 'data_nasabah_baru.json'

if not os.path.exists(filename):
    print(f"Error: File {filename} tidak ditemukan! Jalankan kode generator data dulu.")
else:
    with open(filename, 'r') as f:
        data_nasabah = json.load(f)

    # 2. Siapkan payload sesuai format yang diminta server
    payload = {"data_nasabah_baru": data_nasabah}

    print(f"Mengirim {len(data_nasabah)} data nasabah ke server...")

    # 3. Kirim Request
    try:
        r = requests.post(url, json=payload)
        
        if r.status_code == 200:
            result = r.json()
            print("\nSUKSES!")
            print(f"Hasil Prediksi: {result['prediksi_churn']}")
            
            # (Opsional) Hitung berapa yang diprediksi Churn (1)
            churn_count = sum(result['prediksi_churn'])
            print(f"\nDari {len(data_nasabah)} nasabah baru, diprediksi {churn_count} orang akan Churn.")
        else:
            print("\nGAGAL.")
            print("Status:", r.status_code)
            print("Response:", r.text)
            
    except Exception as e:
        print("Gagal koneksi ke server. Pastikan app.py sudah jalan.", e)