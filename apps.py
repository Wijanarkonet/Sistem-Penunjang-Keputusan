import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import streamlit as st

# ============================================================
# CONFIG & STYLE (STREAMLIT)
# ============================================================
st.set_page_config(
    page_title="TOPSIS - Analisis Treatment Greyclean",
    page_icon="🏆",
    layout="wide"
)

warnings.filterwarnings('ignore')

# Set visual style untuk matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 11

# ============================================================
# UI HEADER
# ============================================================
st.title("🏆 Sistem Penunjang Keputusan — TOPSIS")
st.subheader("Analisis Treatment Paling Populer | GREYCLEAN 2026")
st.markdown("---")

# ============================================================
# SIDEBAR - UPLOAD & INPUT
# ============================================================
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload File Excel Transaksi", type=["xlsx", "xls"])

st.sidebar.header("⚙️ Pengaturan Bobot Kriteria")
b1 = st.sidebar.slider("C1 - Jumlah Transaksi", 0.0, 1.0, 0.35, 0.05)
b2 = st.sidebar.slider("C2 - Total Pendapatan", 0.0, 1.0, 0.35, 0.05)
b3 = st.sidebar.slider("C3 - Pelanggan Unik", 0.0, 1.0, 0.20, 0.05)
b4 = st.sidebar.slider("C4 - Rata-rata Harga", 0.0, 1.0, 0.10, 0.05)

# Normalisasi bobot jika totalnya tidak sama dengan 1
total_bobot = b1 + b2 + b3 + b4
if total_bobot == 0:
    st.error("Total bobot tidak boleh 0!")
    st.stop()
bobot = np.array([b1, b2, b3, b4]) / total_bobot

# Main logical block jika file sudah di-upload
if uploaded_file is not None:
    try:
        xl = pd.ExcelFile(uploaded_file)
        
        # Menampilkan status sheet di sidebar
        st.sidebar.success(f"Loaded: {len(xl.sheet_names)} Bulan/Sheet")
        
        all_data = []
        for sheet in xl.sheet_names:
            df_raw = pd.read_excel(xl, sheet_name=sheet, header=None)
            df = df_raw.iloc[5:].copy()
            
            df.columns = [
                'No_Transaksi', 'Tanggal', 'Nama_Pelanggan', 'No_HP',
                'Metode_Pembayaran', 'Nama_Barang', 'Tipe_Treatment', 'Total_Harga'
            ]
            df['Bulan'] = sheet
            
            # Filter baris yang memiliki No_Transaksi berupa angka valid
            df = df[df['No_Transaksi'].apply(
                lambda x: str(x).strip().replace('.0', '').isdigit() if pd.notna(x) else False
            )]
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)
        df_all['Total_Harga'] = pd.to_numeric(df_all['Total_Harga'], errors='coerce')
        df_all.dropna(subset=['Total_Harga'], inplace=True)
        
        df_all['Tipe_Treatment'] = (
            df_all['Tipe_Treatment'].astype(str).str.strip().str.title()
        )
        df_all = df_all[~df_all['Tipe_Treatment'].isin(['-', 'Nan', 'None', ''])]

        # ============================================================
        # TABS INTERFACE
        # ============================================================
        tab1, tab2, tab3 = st.tabs(["📊 Data & Matriks", "🧮 Proses TOPSIS", "🏆 Hasil & Visualisasi"])

        with tab1:
            st.metric("Total Seluruh Transaksi", f"{len(df_all)} Transaksi")
            st.subheader("1. Matriks Keputusan Awal (X)")
            
            df_matrix = df_all.groupby('Tipe_Treatment').agg(
                C1_Jumlah_Transaksi=('No_Transaksi', 'count'),
                C2_Total_Pendapatan=('Total_Harga', 'sum'),
                C3_Pelanggan_Unik=('Nama_Pelanggan', 'nunique'),
                C4_Rata_Rata_Harga=('Total_Harga', 'mean')
            ).reset_index()
            
            df_matrix = df_matrix.sort_values('C1_Jumlah_Transaksi', ascending=False)
            st.dataframe(df_matrix.style.format({
                'C2_Total_Pendapatan': 'Rp {:,.2f}',
                'C4_Rata_Rata_Harga': 'Rp {:,.2f}'
            }), use_container_width=True)

        with tab2:
            st.subheader("2. Proses Normalisasi & Pembobotan")
            
            X = df_matrix[['C1_Jumlah_Transaksi', 'C2_Total_Pendapatan', 'C3_Pelanggan_Unik', 'C4_Rata_Rata_Harga']].values.astype(float)
            penyebut = np.sqrt((X**2).sum(axis=0))
            
            # Cegah pembagian dengan nol
            penyebut = np.where(penyebut == 0, 1, penyebut)
            
            R = X / penyebut
            df_R = pd.DataFrame(R, columns=['C1 (Transaksi)', 'C2 (Pendapatan)', 'C3 (Pelanggan)', 'C4 (Rata-rata Harga)'], index=df_matrix['Tipe_Treatment'])
            
            st.markdown("**Matriks Ternormalisasi (R):**")
            st.dataframe(df_R.style.format("{:.4f}"), use_container_width=True)
            
            V = R * bobot
            df_V = pd.DataFrame(V, columns=['C1', 'C2', 'C3', 'C4'], index=df_matrix['Tipe_Treatment'])
            
            st.markdown("**Matriks Ternormalisasi Terbobot (V):**")
            st.dataframe(df_V.style.format("{:.4f}"), use_container_width=True)

            # Solusi Ideal
            A_plus = V.max(axis=0)
            A_minus = V.min(axis=0)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Solusi Ideal Positif (A+):**")
                st.write(pd.DataFrame([A_plus], columns=['C1', 'C2', 'C3', 'C4'], index=['Max']))
            with col_b:
                st.markdown("**Solusi Ideal Negatif (A-):**")
                st.write(pd.DataFrame([A_minus], columns=['C1', 'C2', 'C3', 'C4'], index=['Min']))

            # Jarak Solusi Ideal
            D_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
            D_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
            
            df_jarak = pd.DataFrame({
                'Treatment': df_matrix['Tipe_Treatment'],
                'D+ (Jarak Positif)': D_plus,
                'D- (Jarak Negatif)': D_minus
            })
            st.markdown("**Jarak Solusi Ideal (D+ dan D-):**")
            st.dataframe(df_jarak.style.format({'D+ (Jarak Positif)': '{:.4f}', 'D- (Jarak Negatif)': '{:.4f}'}), use_container_width=True)

        with tab3:
            # Perhitungan Kedekatan Kedekatan (Ci)
            Ci = D_minus / (D_plus + D_minus)
            
            df_hasil = pd.DataFrame({
                'Treatment': df_matrix['Tipe_Treatment'],
                'Jumlah_Transaksi': df_matrix['C1_Jumlah_Transaksi'],
                'Total_Pendapatan': df_matrix['C2_Total_Pendapatan'],
                'Pelanggan_Unik': df_matrix['C3_Pelanggan_Unik'],
                'Nilai_Preferensi': Ci
            })
            df_hasil['Ranking'] = df_hasil['Nilai_Preferensi'].rank(ascending=False).astype(int)
            df_hasil = df_hasil.sort_values('Ranking')

            # Tampilkan Kesimpulan Utama di Atas Tab Hasil
            terbaik = df_hasil.iloc[0]
            st.success(f"""
            ### 🏆 KESIMPULAN ANALISIS (ALTERNATIF TERBAIK)
            Treatment paling populer berdasarkan metode TOPSIS adalah **{terbaik['Treatment']}**.
            
            * **Nilai Preferensi (Ci):** `{terbaik['Nilai_Preferensi']:.4f}`
            * **Jumlah Transaksi:** `{terbaik['Jumlah_Transaksi']} kali`
            * **Total Pendapatan:** `Rp {terbaik['Total_Pendapatan']:,.0f}`
            * **Pelanggan Unik:** `{terbaik['Pelanggan_Unik']} orang`
            
            *Treatment tersebut menjadi alternatif terbaik karena memiliki performa indeks preferensi tertinggi mendekati solusi ideal positif.*
            """)
            
            st.markdown("---")
            st.subheader("Tabel Peringkat Akhir")
            st.dataframe(df_hasil.style.format({
                'Total_Pendapatan': 'Rp {:,.2f}',
                'Nilai_Preferensi': '{:.4f}'
            }), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Visualisasi Grafik")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.barh(df_hasil['Treatment'][::-1], df_hasil['Nilai_Preferensi'][::-1], color='royalblue')
                ax1.set_xlabel('Nilai Preferensi')
                ax1.set_ylabel('Treatment')
                ax1.set_title('Ranking Treatment Berdasarkan Nilai Preferensi')
                for i, v in enumerate(df_hasil['Nilai_Preferensi'][::-1]):
                    ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig1)

            with col_g2:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(df_hasil['Treatment'][::-1], df_hasil['Jumlah_Transaksi'][::-1], color='teal')
                ax2.set_xlabel('Jumlah Transaksi')
                ax2.set_ylabel('Treatment')
                ax2.set_title('Perbandingan Volume Transaksi per Treatment')
                plt.tight_layout()
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

else:
    st.info("💡 Silakan unggah file Excel data transaksi Greyclean pada sidebar untuk memulai kalkulasi TOPSIS.")