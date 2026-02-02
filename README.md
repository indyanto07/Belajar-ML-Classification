# Bank Customer Churn Prediction 🏦

Proyek ini bertujuan untuk memprediksi kemungkinan nasabah meninggalkan bank (**Churn**) menggunakan algoritma **Support Vector Machine (SVM)**. Dilengkapi dengan API berbasis **Flask** untuk melakukan prediksi secara *real-time* terhadap data nasabah baru.

## 📁 Struktur Proyek

* `app.py`: Backend Flask API yang melayani request prediksi.
* `SVM_model.joblib`: Model SVC (Support Vector Classifier) yang sudah dilatih dengan `class_weight='balanced'`.
* `scaler.joblib`: File `MinMaxScaler` untuk normalisasi fitur numerik.
* `xyz.ipynb`: Notebook utama proses EDA, preprocessing, hingga training model.
* `data_nasabah_baru.json`: Contoh data input format JSON untuk pengujian API.
* `test_predict.py`: Script Python untuk menguji endpoint API secara otomatis.
* `requirements.txt`: Daftar pustaka (library) yang diperlukan.

## 📊 Analisis Performa Model

Dalam proyek ini, model **SVM** dipilih dan dioptimalkan untuk menangani *imbalanced dataset* (ketidakseimbangan jumlah nasabah churn vs tidak churn).

### Metrik Evaluasi (SVM Balanced)

Dengan menggunakan konfigurasi `class_weight='balanced'`, model memberikan prioritas lebih pada kelas minoritas (Nasabah Churn).

| Metrik              | Skor     | Catatan                                                                                  |
| **Accuracy**        | **~81%** | Kemampuan klasifikasi total data secara tepat.                                           |
| **Recall (Churn)**  | **73%**  | **Fokus Utama**: Model berhasil mendeteksi 73% dari total nasabah yang sebenarnya churn. |
| **Precision**       | **~52%** | Ketepatan prediksi ketika model menandai nasabah sebagai churn.                          |
| **Kernel**          | **RBF**  | Digunakan untuk menangani hubungan non-linear antar fitur.                               |

> **Mengapa Recall Tinggi Itu Penting?** > Dalam industri perbankan, lebih baik kita "salah mencurigai" nasabah akan churn (False Positive) agar bisa diberikan penawaran retensi, daripada "gagal mendeteksi" nasabah yang benar-benar akan pergi (False Negative). Oleh karena itu, **Recall 73%** jauh lebih berharga daripada akurasi tinggi namun gagal mendeteksi churn.

## 🛠️ Instalasi & Persiapan

1. **Clone Repositori**

git clone https://github.com/username/bank-churn-prediction.git
cd bank-churn-prediction

2. **Install Library**

pip install -r requirements.txt


## 🚀 Cara Menjalankan

### 1. Jalankan API Server


python app.py



Server akan aktif di `http://127.0.0.1:5000`.

### 2. Lakukan Prediksi

Gunakan script tester yang sudah disediakan untuk mengirim data dari file JSON:


python test_predict.py



### 3. Contoh Request JSON

API menerima input dengan format berikut:

```json
{
  "data_nasabah_baru": [
    {
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0.0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88
    }
  ]
}

```

## ⚙️ Alur Kerja (Pipeline)

1. **Preprocessing**: Data kategori (`Geography`, `Gender`) diubah menjadi numerik secara manual di dalam API untuk konsistensi.
2. **Scaling**: Fitur numerik diproses menggunakan `MinMaxScaler` agar berada dalam rentang  sampai  sesuai kebutuhan algoritma SVM.
3. **Prediction**: Model memproses data yang sudah di-scale dan mengembalikan label `0` (Stay) atau `1` (Churn).

