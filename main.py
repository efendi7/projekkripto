import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


# Fungsi Hamming Weight
def hamming_weight(x):
    return bin(x).count("1")


# Fungsi untuk menghitung Nonlinearity (NL)
def calculate_nonlinearity_boolean(func):
    n = int(np.log2(len(func)))
    walsh_transform = []
    for mask in range(1 << n):
        walsh_sum = sum(
            (-1) ** ((hamming_weight(x & mask) ^ func[x])) for x in range(len(func))
        )
        walsh_transform.append(abs(walsh_sum))
    nl = (2 ** (n - 1)) - max(walsh_transform) / 2
    return nl


def calculate_nl_matrix(sbox):
    n = int(np.log2(len(sbox)))
    nl_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            func = [(sbox[x] >> j) & 1 for x in range(len(sbox))]
            nl_matrix[i, j] = calculate_nonlinearity_boolean(func)
    return nl_matrix


# Fungsi untuk menghitung SAC Matrix
def calculate_sac_matrix(sbox):
    n = int(np.log2(len(sbox)))
    sac_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diffs = [(sbox[x] ^ sbox[x ^ (1 << i)]) >> j & 1 for x in range(len(sbox))]
            sac_matrix[i, j] = sum(diffs) / len(sbox)
    return sac_matrix


# Fungsi untuk menghitung LAP
def calculate_lap_matrix(sbox):
    n = int(np.log2(len(sbox)))  # Menghitung jumlah bit
    lap_matrix = np.zeros((n, n))  # Matriks untuk LAP

    # Iterasi untuk setiap bit input (i) dan bit output (j)
    for i in range(n):
        for j in range(n):
            valid_approx = 0  # Hitung aproksimasi valid
            total_approx = 0  # Hitung total aproksimasi

            # Loop untuk setiap input pada sbox
            for x in range(len(sbox)):
                # Aproksimasi linier berdasarkan bit input i dan output j
                approx = (x >> i) ^ (sbox[x] >> j)
                if approx == 0:  # Jika aproksimasi valid
                    valid_approx += 1
                total_approx += 1  # Semua aproksimasi diuji

            # Probabilitas LAP
            lap_matrix[i, j] = valid_approx / total_approx

    return lap_matrix


# Fungsi untuk menghitung DAP
def calculate_dap_matrix(sbox):
    n = int(np.log2(len(sbox)))
    dap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diffs = [0] * (1 << n)
            for x in range(len(sbox)):
                diff_input = x ^ (1 << i)
                diff_output = sbox[x] ^ sbox[diff_input]
                diffs[diff_output] += 1
            dap_matrix[i, j] = max(diffs) / len(sbox)
    return dap_matrix


# Fungsi BIC-SAC Matrix
def calculate_bic_sac_matrix(sbox):
    n = int(np.log2(len(sbox)))  # Menghitung jumlah bit berdasarkan panjang S-box
    bic_sac_matrix = np.zeros((n, n))  # Matriks hasil BIC-SAC

    for i in range(n):  # Iterasi untuk setiap bit masukan
        for j in range(n):  # Iterasi untuk bit keluaran pertama
            diffs_j = [
                (sbox[x] ^ sbox[x ^ (1 << i)]) >> j & 1 for x in range(len(sbox))
            ]
            for k in range(n):  # Iterasi untuk bit keluaran kedua (berbeda dari j)
                if j != k:  # Pastikan hanya menghitung antar bit keluaran berbeda
                    diffs_k = [
                        (sbox[x] ^ sbox[x ^ (1 << i)]) >> k & 1
                        for x in range(len(sbox))
                    ]
                    # Hitung proporsi elemen yang sama antara dua bit keluaran
                    similarity = sum(dj == dk for dj, dk in zip(diffs_j, diffs_k))
                    bic_sac_matrix[j, k] += similarity / len(sbox)

    bic_sac_matrix /= n  # Normalisasi terhadap jumlah bit masukan

    # Membulatkan hasil ke 6 angka desimal jika diperlukan untuk konsistensi
    bic_sac_matrix = np.round(bic_sac_matrix, 6)

    return bic_sac_matrix


# Fungsi untuk menghitung BIC-NL
def calculate_bic_nl_matrix(sbox):
    n = int(np.log2(len(sbox)))
    bic_nl_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            func = [(sbox[x] >> j) & 1 for x in range(len(sbox))]
            bic_nl_matrix[i, j] = calculate_nonlinearity_boolean(func)
    return bic_nl_matrix


# Aplikasi Streamlit
st.title("S-Box Cryptographic Strength Testing")

# Subjudul Anggota Tim
st.subheader("Anggota Kelompok:")
st.markdown(
    """
1. Aditya Christian Nugroho (4611422050)  
2. Abi Kurniawan (4611422070)  
3. Sadya Putra Nuring Bawono (4611422073)  
4. Muhammad Ma'mun Efendi (4611422081)  
"""
)

# Upload File
uploaded_file = st.file_uploader(
    "Upload S-Box File (Excel/Spreadsheet)", type=["xlsx", "xls"]
)
if uploaded_file:
    sbox_df = pd.read_excel(uploaded_file, header=None)
    sbox_array = sbox_df.values.flatten()
    st.write("### S-Box Matrix")
    st.dataframe(sbox_df)

    # Pilihan Operasi
    operation = st.selectbox(
        "Pilih Operasi Pengujian",
        [
            "Nonlinearity (NL)",
            "Strict Avalanche Criterion (SAC)",
            "Linear Approximation Probability (LAP)",
            "Differential Approximation Probability (DAP)",
            "Bit Independence Criterion - SAC (BIC-SAC)",
            "Bit Independence Criterion - NL (BIC-NL)",
        ],
    )

    # Hitung berdasarkan operasi yang dipilih
    if st.button("Hitung"):
        if operation == "Nonlinearity (NL)":
            nl_matrix = calculate_nl_matrix(sbox_array)
            nl_matrix = nl_matrix[:, ::-1]  # Membalik kolom NL Matrix
            st.write("### NL Matrix")
            st.dataframe(pd.DataFrame(nl_matrix))
            avg_nl = np.mean(nl_matrix)
            st.write(f"### Rata-rata NL: {avg_nl:.6f}")

            buffer = BytesIO()
            nl_df = pd.DataFrame(nl_matrix)
            nl_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download NL Matrix Excel",
                data=buffer,
                file_name="nl_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elif operation == "Strict Avalanche Criterion (SAC)":
            sac_matrix = calculate_sac_matrix(sbox_array)
            sac_matrix = sac_matrix[:, ::-1]  # Membalik kolom SAC Matrix
            st.write("### SAC Matrix")
            st.dataframe(pd.DataFrame(sac_matrix))

            avg_sac = np.mean(sac_matrix)
            st.write(f"### Rata-rata SAC: {avg_sac:.6f}")

            buffer = BytesIO()
            sac_df = pd.DataFrame(sac_matrix)
            sac_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download SAC Matrix Excel",
                data=buffer,
                file_name="sac_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elif operation == "Linear Approximation Probability (LAP)":
            lap_matrix = calculate_lap_matrix(sbox_array)
            st.write("### LAP Matrix")
            st.dataframe(pd.DataFrame(lap_matrix))
            avg_lap = np.mean(lap_matrix)
            st.write(f"### Rata-rata LAP: {avg_lap:.6f}")

            buffer = BytesIO()
            lap_df = pd.DataFrame(lap_matrix)
            lap_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download LAP Matrix Excel",
                data=buffer,
                file_name="lap_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elif operation == "Differential Approximation Probability (DAP)":
            dap_matrix = calculate_dap_matrix(sbox_array)
            dap_matrix = dap_matrix[:, ::-1]  # Membalik kolom DAP Matrix
            st.write("### DAP Matrix")
            st.dataframe(pd.DataFrame(dap_matrix))
            avg_dap = np.mean(dap_matrix)
            st.write(f"### Rata-rata DAP: {avg_dap:.6f}")

            buffer = BytesIO()
            dap_df = pd.DataFrame(dap_matrix)
            dap_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download DAP Matrix Excel",
                data=buffer,
                file_name="dap_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elif operation == "Bit Independence Criterion - SAC (BIC-SAC)":
            bic_sac_matrix = calculate_bic_sac_matrix(sbox_array)
            st.write("### BIC-SAC Matrix")
            st.dataframe(pd.DataFrame(bic_sac_matrix))
            avg_bic_sac = np.mean(bic_sac_matrix)
            st.write(f"### Rata-rata BIC-SAC: {avg_bic_sac:.6f}")

            buffer = BytesIO()
            bic_sac_df = pd.DataFrame(bic_sac_matrix)
            bic_sac_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download BIC-SAC Matrix Excel",
                data=buffer,
                file_name="bic_sac_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elif operation == "Bit Independence Criterion - NL (BIC-NL)":
            bic_nl_matrix = calculate_bic_nl_matrix(sbox_array)
            bic_nl_matrix = bic_nl_matrix[:, ::-1]  # Membalik kolom BIC-NL Matrix
            st.write("### BIC-NL Matrix")
            st.dataframe(pd.DataFrame(bic_nl_matrix))
            avg_bic_nl = np.mean(bic_nl_matrix)
            st.write(f"### Rata-rata BIC-NL: {avg_bic_nl:.6f}")

            buffer = BytesIO()
            bic_nl_df = pd.DataFrame(bic_nl_matrix)
            bic_nl_df.to_excel(buffer, index=False, header=False)
            buffer.seek(0)

            st.download_button(
                label="Download BIC-NL Matrix Excel",
                data=buffer,
                file_name="bic_nl_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
