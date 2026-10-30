import numpy as np

def metode_power(A, x0, tol=1e-6, max_iter=100):
    """
    Menghitung Nilai Eigen Dominan dan Vektor Eigen menggunakan Metode Power.
    
    Parameters:
        A        : np.ndarray (n x n) -> Matriks persegi
        x0       : np.ndarray (n,)   -> Vektor tebakan awal
        tol      : float             -> Toleransi konvergensi (threshold)
        max_iter : int               -> Batas maksimum iterasi
    """
    x = x0.astype(float)
    nilai_eigen_lama = 0.0
    
    for i in range(max_iter):
        # 1. Perkalian matriks A dengan vektor x_k (y = A * x)
        y = np.dot(A, x)
        
        # 2. Mencari elemen dengan nilai mutlak terbesar sebagai faktor skala
        indeks_max = np.argmax(np.abs(y))
        nilai_eigen_baru = y[indeks_max]  # Mempertahankan tanda positif/negatif
        
        # 3. Normalisasi vektor eigen (x_k+1 = y / scaling_factor)
        x = y / y[indeks_max]
        
        # 4. Cek Konvergensi berdasarkan selisih threshold
        if np.abs(nilai_eigen_baru - nilai_eigen_lama) < tol:
            return nilai_eigen_baru, x, i + 1, True
        
        nilai_eigen_lama = nilai_eigen_baru
        
    return nilai_eigen_baru, x, max_iter, False

def kuosien_rayleigh(A, x):
    """
    Menghitung aproksimasi nilai eigen menggunakan Kuosien Rayleigh:
    lambda = (A * x . x) / (x . x)
    """
    Ax = np.dot(A, x)
    return np.dot(Ax, x) / np.dot(x, x)


if __name__ == "__main__":
    A1 = np.array([[2.0, 1.0], 
                   [1.0, 2.0]])
    
    A2 = np.array([[2.0, -12.0], 
                   [1.0, -5.0]])
    
    print("1. PENGUJIEN UTAMA METODE POWER (Matriks A2)")
    x0_default = np.array([1.0, 1.0])
    lambda_dom, v_dom, iters, status = metode_power(A2, x0_default, tol=1e-6)
    
    if status:
        print(f"Konvergen pada iterasi ke-{iters}")
        print(f"Nilai Eigen Dominan (Skalar Max) : {lambda_dom:.6f}")
        print(f"Kuosien Rayleigh                 : {kuosien_rayleigh(A2, v_dom):.6f}")
        print(f"Vektor Eigen (Terskala)          : {v_dom}")
        
        # Normalisasi Euclidean L2 untuk pembandingan standar
        v_l2 = v_dom / np.linalg.norm(v_dom)
        print(f"Vektor Eigen (Norma L2)          : {v_l2}")

    # Bandingkan dengan NumPy Linalg
    val_exact, vec_exact = np.linalg.eig(A2)
    print("\n--- Hasil Eksak (np.linalg.eig) ---")
    print(f"Nilai Eigen Eksak : {val_exact}")
    print("Vektor Eigen Eksak:\n", vec_exact)

    print("\n2. EKSPERIMEN 1: PENGARUH PERBEDAAN VEKTOR AWAL (x0)")
    vektor_awal_test = [
        np.array([1.0, 1.0]),
        np.array([10.0, 1.0]),
        np.array([-5.0, 3.0]),
        np.array([1.0, 0.0]),
        np.array([4.0, 1.0]) 
    ]
    
    for idx, x_init in enumerate(vektor_awal_test, 1):
        val, vec, num_iter, _ = metode_power(A2, x_init, tol=1e-6)
        # Perbaikan pada baris di bawah ini: mengonversi x_init ke string terlebih dahulu
        str_x_init = str(x_init.tolist())
        print(f"Uji {idx} | x0 = {str_x_init:<12} -> Iterasi: {num_iter:2d} | Nilai Eigen: {val:.6f} | Vektor: {vec}")

    print("\n3. EKSPERIMEN 2: PENGARUH PERBEDAAN THRESHOLD (TOLERANSI)")
    toleransi_test = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    
    for tol in toleransi_test:
        val, vec, num_iter, _ = metode_power(A2, x0_default, tol=tol)
        print(f"Threshold = {tol:.0e} -> Iterasi: {num_iter:2d} | Nilai Eigen: {val:.10f}")
