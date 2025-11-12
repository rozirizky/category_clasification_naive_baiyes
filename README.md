# 🧠 Klasifikasi Kategori Berita dengan Naive Bayes & PySpark

Proyek ini bertujuan untuk **membangun model klasifikasi teks** yang dapat mengelompokkan **judul berita** ke dalam beberapa kategori (seperti *politik, ekonomi, olahraga, kesehatan*, dll) menggunakan **algoritma Naive Bayes** dan **framework Apache Spark** untuk pemrosesan data skala besar.

---

## 🚀 Tujuan Proyek
Model ini dibuat untuk:
- Mengelompokkan berita berdasarkan *judul* secara otomatis.
- Melatih dan menyimpan model dengan pipeline *text preprocessing* lengkap.
- Mengunggah model hasil training ke **Hugging Face Hub** untuk pemanfaatan lebih lanjut (misalnya di aplikasi prediksi atau API NLP).

---

## 🧩 Arsitektur Pipeline

Pipeline yang digunakan dalam proyek ini terdiri dari tahapan berikut:

1. **Tokenizer** — memecah kalimat menjadi token kata.  
2. **StopWordsRemover** — menghapus kata umum (stopwords) Bahasa Indonesia.  
3. **Stemming (Sastrawi)** — mengubah kata ke bentuk dasarnya.  
4. **HashingTF & IDF** — konversi teks ke representasi numerik berbasis frekuensi.  
5. **Naive Bayes Classifier** — model klasifikasi teks probabilistik.  
6. **Label Decoder (IndexToString)** — mengubah hasil prediksi numerik menjadi label kategori asli.

---

## 🛠️ Teknologi yang Digunakan
| Komponen | Deskripsi |
|-----------|------------|
| 🐍 Python | Bahasa pemrograman utama |
| ⚡ Apache Spark (PySpark) | Framework untuk pemrosesan dan ML skala besar |
| 📦 Sastrawi | Library stemming Bahasa Indonesia |
| 🧠 NLTK | Stopword Bahasa Indonesia |
| 🤗 Hugging Face Hub | Tempat penyimpanan dan berbagi model |
| 🔐 dotenv | Mengelola environment variable & token |

---

## 📂 Struktur Proyek
