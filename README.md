# Eigen - Komputasi Numerik

Repository ini berisi implementasi algoritma numerik untuk menghitung **Nilai Eigen Dominan** dan **Vektor Eigen** menggunakan **Metode Power (Power Iteration)** serta **Kuosien Rayleigh (Rayleigh Quotient)** menggunakan Python dan pustaka NumPy.

Repository ini dibuat untuk memenuhi tugas mata kuliah **Komputasi Numerik**.

---

## 📂 Struktur File

* [`tugas_eigen_komnum.py`](tugas_eigen_komnum.py): Script utama yang mencakup implementasi fungsi Metode Power, Kuosien Rayleigh, serta eksperimen pengujian terhadap variasi vektor awal dan toleransi (threshold).

---

## ⚙️ Fitur & Fungsi Utama

1. **`metode_power(A, x0, tol=1e-6, max_iter=100)`**
   * Menghitung nilai eigen dominan dan vektor eigen dari sebuah matriks persegi `A` menggunakan metode iterasi *power*.
   * Mengembalikan nilai eigen baru, vektor eigen terskala, jumlah iterasi yang ditempuh, dan status konvergensi (`True`/`False`).

2. **`kuosien_rayleigh(A, x)`**
   * Menghitung aproksimasi nilai eigen yang lebih akurat dari vektor eigen hasil iterasi menggunakan formula Kuosien Rayleigh: 
     $$\lambda = \frac{Ax \cdot x}{x \cdot x}$$

---

## 🧪 Skenario Eksperimen dalam Script

Script ini dilengkapi dengan beberapa skenario pengujian menggunakan matriks uji $A_2$:
* **Pengujian Utama:** Eksekusi Metode Power pada matriks $A_2$, perbandingan hasil aproksimasi dengan Kuosien Rayleigh, normalisasi Euclidean L2, serta perbandingan langsung dengan hasil eksak bawaan pustaka (`np.linalg.eig`).
* **Eksperimen 1:** Analisis pengaruh perbedaan tebakan vektor awal ($x_0$) terhadap kecepatan konvergensi dan hasil akhir iterasi.
* **Eksperimen 2:** Analisis pengaruh tingkat toleransi (*threshold*) terhadap ketelitian nilai eigen yang dihasilkan.

---

## 🚀 Cara Menjalankan Program

Pastikan kamu sudah menginstal pustaka **NumPy** di lingkungan Python kamu. Jika belum, instal menggunakan perintah berikut:

```bash
pip install numpy
