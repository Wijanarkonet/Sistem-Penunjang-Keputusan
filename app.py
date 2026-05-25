# ============================================================
# app.py — Streamlit Web App
# Sistem Pendukung Keputusan TOPSIS
# Penentuan Pelanggan Terbaik Greyclean 2026
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="SPK TOPSIS — Greyclean 2026",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .step-header {
        background: #f0f4ff;
        border-left: 5px solid #667eea;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        font-weight: 700;
        font-size: 1rem;
        color: #1a1a2e;
        margin: 1rem 0 0.5rem 0;
    }
    .winner-box {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: #1a1a2e;
    }
    .winner-name { font-size: 2rem; font-weight: 800; }
    .winner-ci   { font-size: 1.1rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🧹 Sistem Pendukung Keputusan — Metode TOPSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Penentuan Pelanggan Terbaik • Greyclean 2026 • Januari – April</div>', unsafe_allow_html=True)
st.divider()

# ============================================================
# SIDEBAR — UPLOAD & BOBOT
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/washing-machine.png", width=80)
    st.markdown("## ⚙️ Pengaturan")
    st.divider()

    st.markdown("### 📂 Upload File Excel")
    uploaded_file = st.file_uploader(
        "Upload Rekap_Transaksi_Greyclean_2026.xlsx",
        type=["xlsx"],
        help="File Excel dengan sheet: Januari, Febuari, Maret, April 2026"
    )

    st.divider()
    st.markdown("### ⚖️ Bobot Kriteria")
    st.caption("Geser untuk ubah bobot. Total harus = 1.0")

    w1 = st.slider("C1 — Total Transaksi",    0.05, 0.60, 0.30, 0.05)
    w2 = st.slider("C2 — Total Pengeluaran",  0.05, 0.60, 0.35, 0.05)
    w3 = st.slider("C3 — Rata-rata Harga",    0.05, 0.60, 0.20, 0.05)
    w4 = st.slider("C4 — Variasi Treatment",  0.05, 0.60, 0.15, 0.05)

    total_bobot = round(w1 + w2 + w3 + w4, 2)
    if abs(total_bobot - 1.0) > 0.001:
        st.error(f"❌ Total bobot = {total_bobot:.2f} (harus = 1.00)")
        bobot_valid = False
    else:
        st.success(f"✅ Total bobot = {total_bobot:.2f}")
        bobot_valid = True

    bobot = np.array([w1, w2, w3, w4])

    st.divider()
    st.markdown("### 🔢 Filter Pelanggan")
    min_transaksi = st.selectbox(
        "Minimal jumlah transaksi",
        options=[2, 3, 4],
        index=0,
        help="Pelanggan dengan transaksi di bawah nilai ini tidak dianalisis"
    )

    st.divider()
    st.caption("📌 SPK TOPSIS • Greyclean 2026")

# ============================================================
# FUNGSI PREPROCESSING
# ============================================================
@st.cache_data
def load_and_preprocess(file, min_trx):
    xl = pd.ExcelFile(file)
    all_data = []
    sheet_info = {}

    for sheet in xl.sheet_names:
        df_raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        df = df_raw.iloc[5:].copy()
        df.columns = ['No_Transaksi','Tanggal','Nama_Pelanggan','No_HP',
                      'Metode_Pembayaran','Nama_Barang','Tipe_Treatment','Total_Harga']
        df['Bulan'] = sheet
        df = df[df['No_Transaksi'].apply(
            lambda x: str(x).strip().replace('.0','').isdigit() if pd.notna(x) else False
        )].copy()
        sheet_info[sheet] = len(df)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['Total_Harga'] = pd.to_numeric(df_all['Total_Harga'], errors='coerce')
    df_all.dropna(subset=['Total_Harga'], inplace=True)
    df_all['Nama_Pelanggan'] = df_all['Nama_Pelanggan'].astype(str).str.strip().str.title()
    df_all = df_all[~df_all['Nama_Pelanggan'].isin(['-','Nan','None',''])]

    frekuensi = df_all.groupby('Nama_Pelanggan').size()
    pelanggan_loyal = frekuensi[frekuensi >= min_trx].index
    df_loyal = df_all[df_all['Nama_Pelanggan'].isin(pelanggan_loyal)].copy()

    return df_all, df_loyal, sheet_info, len(pelanggan_loyal)


def hitung_topsis(df_loyal, bobot):
    # Agregasi
    df_matrix = df_loyal.groupby('Nama_Pelanggan').agg(
        C1_Total_Transaksi   = ('No_Transaksi', 'count'),
        C2_Total_Pengeluaran = ('Total_Harga', 'sum'),
        C3_Rata_Rata_Harga   = ('Total_Harga', 'mean'),
        C4_Variasi_Treatment = ('Tipe_Treatment', lambda x: x.dropna().nunique())
    ).reset_index().sort_values('Nama_Pelanggan').reset_index(drop=True)

    X = df_matrix[['C1_Total_Transaksi','C2_Total_Pengeluaran',
                   'C3_Rata_Rata_Harga','C4_Variasi_Treatment']].values.astype(float)

    # Normalisasi
    penyebut = np.sqrt((X ** 2).sum(axis=0))
    R = X / penyebut

    # Terbobot
    V = R * bobot

    # Solusi ideal
    A_plus  = V.max(axis=0)
    A_minus = V.min(axis=0)

    # Jarak separasi
    D_plus  = np.sqrt(((V - A_plus)  ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    # Nilai preferensi
    Ci = D_minus / (D_plus + D_minus)

    # Hasil akhir
    df_hasil = pd.DataFrame({
        'Nama Pelanggan'        : df_matrix['Nama_Pelanggan'].values,
        'C1 Transaksi'          : df_matrix['C1_Total_Transaksi'].values,
        'C2 Pengeluaran'        : df_matrix['C2_Total_Pengeluaran'].values,
        'C3 Rata-rata'          : df_matrix['C3_Rata_Rata_Harga'].values,
        'C4 Variasi'            : df_matrix['C4_Variasi_Treatment'].values,
        'D+'                    : D_plus,
        'D-'                    : D_minus,
        'Nilai Ci'              : Ci,
        'Ranking'               : Ci.argsort()[::-1].argsort() + 1
    }).sort_values('Ranking').reset_index(drop=True)

    return df_matrix, X, R, V, A_plus, A_minus, D_plus, D_minus, Ci, df_hasil


# ============================================================
# MAIN CONTENT
# ============================================================
if uploaded_file is None:
    # Tampilan saat belum upload file
    st.info("👈 Silakan upload file Excel di sidebar kiri untuk memulai analisis.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📋 Tentang Metode TOPSIS**
        
        TOPSIS *(Technique for Order Preference by Similarity to Ideal Solution)*
        adalah metode pengambilan keputusan multi-kriteria yang memilih alternatif
        terbaik berdasarkan jarak terdekat ke solusi ideal positif dan
        terjauh dari solusi ideal negatif.
        """)
    with col2:
        st.markdown("""
        **📊 Kriteria yang Digunakan**
        
        - **C1** — Total Transaksi *(Benefit)*
        - **C2** — Total Pengeluaran *(Benefit)*
        - **C3** — Rata-rata Harga *(Benefit)*
        - **C4** — Variasi Treatment *(Benefit)*
        """)
    with col3:
        st.markdown("""
        **🚀 Cara Menggunakan**
        
        1. Upload file Excel di sidebar
        2. Atur bobot kriteria sesuai kebutuhan
        3. Pilih minimal transaksi
        4. Hasil analisis otomatis tampil
        """)

elif not bobot_valid:
    st.warning("⚠️ Perbaiki bobot di sidebar terlebih dahulu (total harus = 1.00).")

else:
    # ── Load data ──
    with st.spinner("⏳ Memuat dan memproses data..."):
        df_all, df_loyal, sheet_info, n_loyal = load_and_preprocess(uploaded_file, min_transaksi)

    # ── Hitung TOPSIS ──
    df_matrix, X, R, V, A_plus, A_minus, D_plus, D_minus, Ci, df_hasil = hitung_topsis(df_loyal, bobot)
    nama_list = df_matrix['Nama_Pelanggan'].values

    # ============================================================
    # METRIK RINGKASAN
    # ============================================================
    st.markdown("## 📊 Ringkasan Data")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transaksi", f"{len(df_all)}", help="Semua bulan")
    with c2:
        st.metric("Pelanggan Unik", f"{df_all['Nama_Pelanggan'].nunique()}")
    with c3:
        st.metric(f"Pelanggan ≥{min_transaksi}x", f"{n_loyal} orang", help="Yang masuk analisis TOPSIS")
    with c4:
        total_omzet = df_all['Total_Harga'].sum()
        st.metric("Total Omzet", f"Rp {total_omzet:,.0f}")

    # Transaksi per bulan
    st.markdown("**Transaksi per bulan:**")
    cols = st.columns(len(sheet_info))
    for col, (bulan, jumlah) in zip(cols, sheet_info.items()):
        col.metric(bulan, f"{jumlah} transaksi")

    st.divider()

    # ============================================================
    # TAB NAVIGASI
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data & Matriks",
        "🔢 Proses TOPSIS",
        "🏆 Ranking Akhir",
        "📈 Visualisasi",
        "📝 Kesimpulan"
    ])

    # ──────────────────────────────────────────────────────────
    # TAB 1: DATA & MATRIKS KEPUTUSAN
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="step-header">📂 Data Transaksi Bersih</div>', unsafe_allow_html=True)
        st.caption(f"Menampilkan {len(df_loyal)} transaksi dari {n_loyal} pelanggan loyal")

        col_filter, _ = st.columns([2, 3])
        with col_filter:
            bulan_filter = st.multiselect(
                "Filter bulan:",
                options=df_loyal['Bulan'].unique().tolist(),
                default=df_loyal['Bulan'].unique().tolist()
            )
        df_tampil = df_loyal[df_loyal['Bulan'].isin(bulan_filter)][
            ['No_Transaksi','Bulan','Nama_Pelanggan','Tipe_Treatment','Total_Harga']
        ].copy()
        df_tampil['Total_Harga'] = df_tampil['Total_Harga'].apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(df_tampil, use_container_width=True, height=300)

        st.markdown('<div class="step-header">📊 Matriks Keputusan</div>', unsafe_allow_html=True)
        st.caption("Nilai agregasi per pelanggan — semua kriteria bertipe **Benefit** (semakin besar = semakin baik)")

        df_mk = df_matrix.copy()
        df_mk.index = range(1, len(df_mk)+1)
        df_mk_tampil = df_mk.copy()
        df_mk_tampil['C2_Total_Pengeluaran'] = df_mk_tampil['C2_Total_Pengeluaran'].apply(lambda x: f"Rp {x:,.0f}")
        df_mk_tampil['C3_Rata_Rata_Harga']   = df_mk_tampil['C3_Rata_Rata_Harga'].apply(lambda x: f"Rp {x:,.0f}")
        df_mk_tampil.columns = ['Nama Pelanggan','C1 Total Transaksi',
                                'C2 Total Pengeluaran','C3 Rata-rata Harga','C4 Variasi Treatment']
        st.dataframe(df_mk_tampil, use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 2: PROSES TOPSIS
    # ──────────────────────────────────────────────────────────
    with tab2:
        # Normalisasi
        st.markdown('<div class="step-header">① Matriks Ternormalisasi (R)</div>', unsafe_allow_html=True)
        st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")
        df_R = pd.DataFrame(R, index=nama_list, columns=['C1','C2','C3','C4'])
        df_R.index.name = 'Nama Pelanggan'
        st.dataframe(df_R.round(6), use_container_width=True)

        # Terbobot
        st.markdown('<div class="step-header">② Matriks Ternormalisasi Terbobot (V)</div>', unsafe_allow_html=True)
        st.latex(r"v_{ij} = w_j \times r_{ij}")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Bobot C1", f"{bobot[0]:.2f}")
        col_b2.metric("Bobot C2", f"{bobot[1]:.2f}")
        col_b3.metric("Bobot C3", f"{bobot[2]:.2f}")
        col_b4.metric("Bobot C4", f"{bobot[3]:.2f}")
        df_V = pd.DataFrame(V, index=nama_list, columns=['C1','C2','C3','C4'])
        df_V.index.name = 'Nama Pelanggan'
        st.dataframe(df_V.round(6), use_container_width=True)

        # Solusi ideal
        st.markdown('<div class="step-header">③ Solusi Ideal Positif (A⁺) & Negatif (A⁻)</div>', unsafe_allow_html=True)
        col_ap, col_am = st.columns(2)
        with col_ap:
            st.markdown("**A⁺ — Solusi Ideal Positif (nilai terbaik)**")
            df_ap = pd.DataFrame({'Kriteria':['C1','C2','C3','C4'], 'Nilai A⁺': A_plus,
                                  'Dimiliki oleh': [nama_list[V[:,j].argmax()] for j in range(4)]})
            st.dataframe(df_ap, use_container_width=True, hide_index=True)
        with col_am:
            st.markdown("**A⁻ — Solusi Ideal Negatif (nilai terburuk)**")
            df_am = pd.DataFrame({'Kriteria':['C1','C2','C3','C4'], 'Nilai A⁻': A_minus,
                                  'Dimiliki oleh': [nama_list[V[:,j].argmin()] for j in range(4)]})
            st.dataframe(df_am, use_container_width=True, hide_index=True)

        # Jarak separasi
        st.markdown('<div class="step-header">④ Jarak Separasi (D⁺ dan D⁻)</div>', unsafe_allow_html=True)
        st.latex(r"D_i^+ = \sqrt{\sum_{j=1}^{n}(v_{ij} - A_j^+)^2} \qquad D_i^- = \sqrt{\sum_{j=1}^{n}(v_{ij} - A_j^-)^2}")
        df_jarak = pd.DataFrame({
            'Nama Pelanggan': nama_list,
            'D⁺ (ke Ideal Positif)': D_plus,
            'D⁻ (ke Ideal Negatif)': D_minus
        })
        df_jarak.index = range(1, len(df_jarak)+1)
        st.dataframe(df_jarak.style.format({'D⁺ (ke Ideal Positif)': '{:.6f}',
                                            'D⁻ (ke Ideal Negatif)': '{:.6f}'}),
                     use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 3: RANKING AKHIR
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="step-header">⑤ Nilai Preferensi & Ranking Akhir</div>', unsafe_allow_html=True)
        st.latex(r"C_i = \frac{D_i^-}{D_i^+ + D_i^-}")
        st.caption("Nilai Ci mendekati 1.0 = pelanggan semakin mendekati kondisi ideal")

        # Tabel ranking
        df_rank_tampil = df_hasil.copy()
        df_rank_tampil.index = range(1, len(df_rank_tampil)+1)
        df_rank_tampil['C2 Pengeluaran'] = df_rank_tampil['C2 Pengeluaran'].apply(lambda x: f"Rp {x:,.0f}")
        df_rank_tampil['C3 Rata-rata']   = df_rank_tampil['C3 Rata-rata'].apply(lambda x: f"Rp {x:,.0f}")

        st.dataframe(
            df_rank_tampil.style
                .format({'Nilai Ci': '{:.4f}', 'D+': '{:.6f}', 'D-': '{:.6f}'})
                .background_gradient(subset=['Nilai Ci'], cmap='RdYlGn'),
            use_container_width=True
        )

        # TOP 3
        st.markdown("### 🏆 Top 3 Pelanggan Terbaik")
        top3 = df_hasil.head(3)
        medali = ["🥇", "🥈", "🥉"]
        cols = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
            with col:
                st.markdown(f"""
                <div style="background: {'linear-gradient(135deg,#f6d365,#fda085)' if i==0 else
                                         'linear-gradient(135deg,#e0e0e0,#bdbdbd)' if i==1 else
                                         'linear-gradient(135deg,#ffcc80,#ffa726)'};
                            padding:1.2rem; border-radius:12px; text-align:center; color:#1a1a2e;">
                    <div style="font-size:2rem">{medali[i]}</div>
                    <div style="font-size:1.3rem; font-weight:800">{row['Nama Pelanggan']}</div>
                    <div style="font-size:1rem; margin-top:0.3rem">Ci = <b>{row['Nilai Ci']:.4f}</b></div>
                    <div style="font-size:0.85rem; margin-top:0.3rem">{int(row['C1 Transaksi'])}x transaksi • Rp {row['C2 Pengeluaran']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 4: VISUALISASI
    # ──────────────────────────────────────────────────────────
    with tab4:
        nama_urut  = df_hasil['Nama Pelanggan'].values
        ci_urut    = df_hasil['Nilai Ci'].values
        harga_urut = df_hasil['C2 Pengeluaran'].values
        freq_urut  = df_hasil['C1 Transaksi'].values

        WARNA_UTAMA = '#2E86AB'
        WARNA_EMAS  = '#F6AE2D'
        WARNA_HIJAU = '#2DC653'
        warna = [WARNA_EMAS if i==0 else WARNA_HIJAU if i<3 else WARNA_UTAMA
                 for i in range(len(nama_urut))]

        # Grafik 1: Nilai Ci
        st.markdown('<div class="step-header">① Ranking TOPSIS — Nilai Preferensi (Ci)</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, max(5, len(nama_urut)*0.4)))
        bars = ax1.barh(range(len(nama_urut)), ci_urut[::-1],
                        color=warna[::-1], edgecolor='white', linewidth=0.5)
        ax1.set_yticks(range(len(nama_urut)))
        ax1.set_yticklabels(nama_urut[::-1], fontsize=9)
        ax1.set_xlabel('Nilai Preferensi (Ci)')
        ax1.set_title('Ranking TOPSIS — Semakin mendekati 1.0 = semakin baik',
                      fontsize=11, fontweight='bold')
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, ci_urut[::-1]):
            ax1.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8)
        legend_items = [mpatches.Patch(color=WARNA_EMAS,  label='Rank 1'),
                        mpatches.Patch(color=WARNA_HIJAU, label='Rank 2-3'),
                        mpatches.Patch(color=WARNA_UTAMA, label='Rank 4+')]
        ax1.legend(handles=legend_items, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        col_g2, col_g3 = st.columns(2)

        # Grafik 2: Total Pengeluaran
        with col_g2:
            st.markdown('<div class="step-header">② Total Pengeluaran</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, max(4, len(nama_urut)*0.38)))
            ax2.barh(range(len(nama_urut)), harga_urut[::-1],
                     color=warna[::-1], edgecolor='white')
            ax2.set_yticks(range(len(nama_urut)))
            ax2.set_yticklabels(nama_urut[::-1], fontsize=8)
            ax2.set_xlabel('Total Pengeluaran (IDR)')
            ax2.set_title('Total Pengeluaran per Pelanggan', fontweight='bold', fontsize=10)
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'Rp {x/1000:.0f}rb'))
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Grafik 3: Frekuensi
        with col_g3:
            st.markdown('<div class="step-header">③ Frekuensi Transaksi</div>', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(6, max(4, len(nama_urut)*0.38)))
            ax3.barh(range(len(nama_urut)), freq_urut[::-1],
                     color=warna[::-1], edgecolor='white')
            ax3.set_yticks(range(len(nama_urut)))
            ax3.set_yticklabels(nama_urut[::-1], fontsize=8)
            ax3.set_xlabel('Jumlah Transaksi')
            ax3.set_title('Frekuensi Transaksi per Pelanggan', fontweight='bold', fontsize=10)
            ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            for bar, val in zip(ax3.patches, freq_urut[::-1]):
                ax3.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                         f'{val}x', va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # Tombol download grafik
        st.divider()
        fig_all, axes = plt.subplots(3, 1, figsize=(12, 16))
        fig_all.suptitle('Analisis TOPSIS — Greyclean 2026', fontsize=14, fontweight='bold')
        axes[0].barh(range(len(nama_urut)), ci_urut[::-1], color=warna[::-1])
        axes[0].set_yticks(range(len(nama_urut)))
        axes[0].set_yticklabels(nama_urut[::-1], fontsize=8)
        axes[0].set_title('Ranking TOPSIS (Ci)')
        axes[1].barh(range(len(nama_urut)), harga_urut[::-1], color=warna[::-1])
        axes[1].set_yticks(range(len(nama_urut)))
        axes[1].set_yticklabels(nama_urut[::-1], fontsize=8)
        axes[1].set_title('Total Pengeluaran')
        axes[2].barh(range(len(nama_urut)), freq_urut[::-1], color=warna[::-1])
        axes[2].set_yticks(range(len(nama_urut)))
        axes[2].set_yticklabels(nama_urut[::-1], fontsize=8)
        axes[2].set_title('Frekuensi Transaksi')
        plt.tight_layout()

        import io
        buf = io.BytesIO()
        fig_all.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button("📥 Download Grafik (PNG)", buf, "TOPSIS_Greyclean_2026.png", "image/png")
        plt.close()

    # ──────────────────────────────────────────────────────────
    # TAB 5: KESIMPULAN
    # ──────────────────────────────────────────────────────────
    with tab5:
        terbaik      = df_hasil.iloc[0]
        nama_terbaik = terbaik['Nama Pelanggan']
        ci_terbaik   = terbaik['Nilai Ci']

        st.markdown("### 🏆 Pelanggan Terbaik")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f6d365,#fda085);
                    padding:1.5rem; border-radius:16px; text-align:center; color:#1a1a2e; margin-bottom:1rem;">
            <div style="font-size:3rem">🥇</div>
            <div style="font-size:2rem; font-weight:800">{nama_terbaik}</div>
            <div style="font-size:1.1rem; margin-top:0.5rem">
                Nilai Preferensi Ci = <b>{ci_terbaik:.4f}</b>
                &nbsp;|&nbsp; {ci_terbaik*100:.1f}% mendekati kondisi ideal
            </div>
            <div style="font-size:0.95rem; margin-top:0.5rem">
                {int(terbaik['C1 Transaksi'])} transaksi &nbsp;•&nbsp;
                Rp {terbaik['C2 Pengeluaran']:,.0f} total pengeluaran &nbsp;•&nbsp;
                {int(terbaik['C4 Variasi'])} variasi treatment
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("#### 📋 Ringkasan Analisis")
            st.markdown(f"""
            - **Total transaksi dianalisis:** {len(df_all)} transaksi
            - **Periode:** Januari – April 2026
            - **Pelanggan dianalisis:** {n_loyal} orang (≥{min_transaksi}x transaksi)
            - **Metode:** TOPSIS dengan 4 kriteria Benefit
            - **Bobot:** C1={bobot[0]:.2f}, C2={bobot[1]:.2f}, C3={bobot[2]:.2f}, C4={bobot[3]:.2f}
            """)
        with col_k2:
            st.markdown("#### 💡 Rekomendasi Bisnis")
            st.markdown(f"""
            1. Berikan **reward / program loyalitas** kepada TOP 3 pelanggan
            2. Jadikan profil **{nama_terbaik}** sebagai acuan pelanggan ideal
            3. Tawarkan **promo cross-selling** kepada pelanggan dengan variasi treatment rendah
            4. Lakukan **analisis ulang setiap kuartal** untuk memantau perubahan perilaku pelanggan
            """)

        st.divider()
        st.markdown("#### ⚠️ Catatan Metodologi")
        st.info(f"""
        Analisis ini hanya mencakup pelanggan dengan **≥{min_transaksi} transaksi** agar penilaian TOPSIS 
        lebih bermakna. Pelanggan dengan transaksi di bawah batas tersebut tidak diikutsertakan 
        karena datanya belum cukup untuk menilai loyalitas secara objektif.
        """)

        # Download hasil Excel
        st.divider()
        st.markdown("#### 💾 Export Hasil")
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_loyal.reset_index(drop=True).to_excel(writer, sheet_name='Data Bersih', index=False)
            df_matrix.to_excel(writer, sheet_name='Matriks Keputusan', index=False)
            pd.DataFrame(R, index=nama_list, columns=['C1','C2','C3','C4']).to_excel(
                writer, sheet_name='Matriks Normalisasi')
            pd.DataFrame(V, index=nama_list, columns=['C1','C2','C3','C4']).to_excel(
                writer, sheet_name='Matriks Terbobot')
            df_hasil.to_excel(writer, sheet_name='Hasil Ranking TOPSIS', index=False)
        output.seek(0)
        st.download_button(
            "📥 Download Hasil Lengkap (Excel)",
            output,
            "Hasil_TOPSIS_Greyclean_2026.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
