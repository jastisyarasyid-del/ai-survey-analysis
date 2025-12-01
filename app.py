import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro
import matplotlib.pyplot as plt

# -----------------------------
# Konfigurasi halaman
# -----------------------------
st.set_page_config(page_title="Analisis Survey AI Tools", layout="wide")

st.title("📊 Analisis Pengaruh Penggunaan AI Tools terhadap Efektivitas Belajar Mahasiswa")

st.write("""
Aplikasi ini akan:
- Membaca file CSV hasil Google Forms mengenai penggunaan AI tools
- Menghitung skor komposit:
  - **X_total** = skor penggunaan AI tools (10 item)
  - **Y_total** = skor efektivitas belajar (10 item)
- Menampilkan statistik deskriptif
- Menguji normalitas (Shapiro-Wilk)
- Menghitung korelasi (Pearson / Spearman) antara X_total dan Y_total
- Menampilkan visualisasi (histogram & scatter plot)
""")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload file CSV (hasil download dari Google Forms)", type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu (misalnya: Survey_ai.csv).")
    st.stop()

# Baca CSV
df = pd.read_csv(uploaded_file)

st.subheader("👀 Preview Data (5 baris pertama)")
st.dataframe(df.head())

# Normalisasi nama kolom (hapus spasi di awal/akhir)
df.columns = df.columns.str.strip()

# -----------------------------
# Definisi kolom item X dan Y
# (disesuaikan dengan pertanyaan di Google Form kamu)
# -----------------------------

# VARIABEL X: Penggunaan AI Tools (10 item)
X_COLS = [
    "Saya menggunakan AI tools untuk membantu memahami materi kuliah.",
    "AI tools membantu saya menyelesaikan tugas lebih cepat.",
    "AI tools membuat saya lebih mudah menemukan penjelasan konsep.",
    "Saya menggunakan AI tools secara rutin saat belajar mandiri.",
    "AI tools membantu saya merangkum materi kuliah.",
    "Saya menggunakan AI tools untuk mendapatkan ide saat mengerjakan tugas.",
    "AI tools membuat proses belajar saya terasa lebih efisien.",
    "Saya merasa lebih percaya diri belajar dengan bantuan AI tools.",
    "AI tools membantu saya memperbaiki kesalahan dalam tugas atau laporan.",
    "Saya merasa kualitas hasil belajar saya meningkat dengan bantuan AI Tools."
]

# VARIABEL Y: Efektivitas Belajar (10 item)
Y_COLS = [
    "Saya mampu memahami materi kuliah dengan baik.",
    "Saya dapat menyelesaikan tugas tepat waktu.",
    "Saya mampu fokus saat belajar.",
    "Metode belajar saya terasa semakin efektif.",
    "Produktivitas belajar saya meningkat.",
    "Saya dapat meninjau materi dengan lebih terstruktur.",
    "Saya mampu mengatur waktu belajar dengan baik.",
    "Saya mampu mengingat materi pembelajaran dengan lebih baik.",
    "Saya dapat menyelesaikan lebih banyak materi dalam waktu yang sama.",
    "Saya merasa hasil belajar saya meningkat secara keseluruhan."
]

# Cek apakah kolom-kolom ini ada di CSV
missing_x = [c for c in X_COLS if c not in df.columns]
missing_y = [c for c in Y_COLS if c not in df.columns]

if missing_x or missing_y:
    st.error(
        "Masih ada kolom yang tidak ditemukan di CSV.\n\n"
        f"Missing kolom X (AI Tools): {missing_x}\n\n"
        f"Missing kolom Y (Efektivitas Belajar): {missing_y}\n\n"
        "Pastikan teks pertanyaan di Google Forms tidak diubah ketika export CSV."
    )
    st.stop()

# -----------------------------
# Konversi ke numerik
# -----------------------------
for c in X_COLS + Y_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Buat dataframe khusus item X & Y
items_df = df[X_COLS + Y_COLS].copy()

# Buang responden yang terlalu banyak missing
items_df["valid_count"] = items_df.notna().sum(axis=1)
df_clean = items_df[items_df["valid_count"] >= 18].copy()  # minimal 18 dari 20 item terisi
df_clean = df_clean.drop(columns=["valid_count"])

st.success(f"Jumlah responden setelah cleaning (min 18/20 item terisi): {len(df_clean)}")

if len(df_clean) == 0:
    st.error("Tidak ada responden yang memenuhi kriteria valid (>=18 item terisi).")
    st.stop()

# -----------------------------
# Hitung skor komposit X & Y
# -----------------------------
df_clean["X_total"] = df_clean[X_COLS].sum(axis=1)
df_clean["Y_total"] = df_clean[Y_COLS].sum(axis=1)
df_clean["X_mean"] = df_clean[X_COLS].mean(axis=1)
df_clean["Y_mean"] = df_clean[Y_COLS].mean(axis=1)

# -----------------------------
# Statistik Deskriptif
# -----------------------------
st.subheader("📌 Statistik Deskriptif — Item-level")
st.write(df_clean[X_COLS + Y_COLS].describe())

st.subheader("📌 Statistik Deskriptif — Skor Komposit (X_total & Y_total)")
st.write(df_clean[["X_total", "Y_total", "X_mean", "Y_mean"]].describe())

# -----------------------------
# Cronbach's Alpha (Reliabilitas)
# -----------------------------
def cronbach_alpha(df_items: pd.DataFrame) -> float:
    df_items = df_items.dropna(axis=0, how="any")
    k = df_items.shape[1]
    if k <= 1 or df_items.shape[0] == 0:
        return np.nan
    item_var = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    return alpha

st.subheader("🧪 Reliabilitas (Cronbach's Alpha)")

alpha_x = cronbach_alpha(df_clean[X_COLS])
alpha_y = cronbach_alpha(df_clean[Y_COLS])

st.write(f"α untuk Penggunaan AI Tools (X): **{alpha_x:.4f}**")
st.write(f"α untuk Efektivitas Belajar (Y): **{alpha_y:.4f}**")

# -----------------------------
# Uji Normalitas Shapiro-Wilk
# -----------------------------
st.subheader("🧪 Uji Normalitas (Shapiro-Wilk) pada X_total dan Y_total")

shapiro_x = shapiro(df_clean["X_total"])
shapiro_y = shapiro(df_clean["Y_total"])

st.write(f"X_total — W = {shapiro_x.statistic:.4f}, p = {shapiro_x.pvalue:.6f}")
st.write(f"Y_total — W = {shapiro_y.statistic:.4f}, p = {shapiro_y.pvalue:.6f}")

if shapiro_x.pvalue > 0.05 and shapiro_y.pvalue > 0.05:
    method = "pearson"
else:
    method = "spearman"

# -----------------------------
# Korelasi X_total vs Y_total
# -----------------------------
st.subheader("📈 Korelasi antara Penggunaan AI (X_total) dan Efektivitas Belajar (Y_total)")

if method == "pearson":
    r, p = pearsonr(df_clean["X_total"], df_clean["Y_total"])
    st.write("Metode korelasi: **Pearson Correlation** (karena data normal).")
else:
    r, p = spearmanr(df_clean["X_total"], df_clean["Y_total"])
    st.write("Metode korelasi: **Spearman Rank Correlation** (karena data tidak normal).")

st.write(f"Koefisien korelasi (r): **{r:.4f}**")
st.write(f"p-value: **{p:.6f}**")

if p < 0.05:
    st.success(
        "Kesimpulan: Terdapat hubungan yang **signifikan** antara penggunaan AI tools "
        "dan efektivitas belajar mahasiswa (p < 0.05)."
    )
else:
    st.warning(
        "Kesimpulan: Tidak terdapat hubungan yang signifikan antara penggunaan AI tools "
        "dan efektivitas belajar mahasiswa (p ≥ 0.05)."
    )

# -----------------------------
# Visualisasi
# -----------------------------
st.subheader("📊 Visualisasi Distribusi dan Hubungan X_total & Y_total")

# Histogram X_total
fig1, ax1 = plt.subplots()
ax1.hist(df_clean["X_total"], bins=10)
ax1.set_title("Histogram X_total (Penggunaan AI Tools)")
ax1.set_xlabel("X_total")
ax1.set_ylabel("Frekuensi")
st.pyplot(fig1)

# Histogram Y_total
fig2, ax2 = plt.subplots()
ax2.hist(df_clean["Y_total"], bins=10)
ax2.set_title("Histogram Y_total (Efektivitas Belajar)")
ax2.set_xlabel("Y_total")
ax2.set_ylabel("Frekuensi")
st.pyplot(fig2)

# Scatter plot X_total vs Y_total
fig3, ax3 = plt.subplots()
ax3.scatter(df_clean["X_total"], df_clean["Y_total"])
ax3.set_title("Scatter Plot: X_total vs Y_total")
ax3.set_xlabel("X_total (Penggunaan AI Tools)")
ax3.set_ylabel("Y_total (Efektivitas Belajar)")
st.pyplot(fig3)

# -----------------------------
# Download data hasil olahan
# -----------------------------
st.subheader("📥 Download Data dengan Skor Komposit")

csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV (data + X_total & Y_total)",
    data=csv_bytes,
    file_name="cleaned_survey_with_composites.csv",
    mime="text/csv"
)
