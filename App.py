import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Prediksi BTC", layout="wide")
st.title("Dashboard Analisis & Prediksi Harga Bitcoin (BTC)")
st.markdown("Eksperimen Regresi Linier Berganda untuk memprediksi harga penutupan (Close) berdasarkan Open, High, Low, dan Volume.")

# --- LOAD & CLEAN DATA ---
@st.cache_data
def load_data():
    # Pastikan file CSV berada di folder yang sama
    df = pd.read_csv('dataset_historis_btc.csv', sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()

# --- BAGIAN 1: RINGKASAN DATA ---
st.header("1. Ringkasan Data Historis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Teratas")
    st.dataframe(df.head())
with col2:
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

st.divider()

# --- PENENTUAN VARIABEL & SPLIT DATA ---
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODELING DENGAN STATSMODELS ---
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()

X_test_sm = sm.add_constant(X_test)
y_pred = model.predict(X_test_sm)
residuals = y_test - y_pred

# --- BAGIAN 2: HASIL PREDIKSI & UJI ASUMSI ---
st.header("2. Hasil Prediksi & Uji Asumsi Klasik")

tab_regresi, tab_vif = st.tabs(["Ringkasan Regresi (OLS)", "Uji Multikolinearitas (VIF)"])

with tab_regresi:
    st.text(model.summary().as_text())

with tab_vif:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    st.dataframe(vif_data)
    st.info("Nilai VIF > 10 mengindikasikan adanya multikolinearitas tinggi antar variabel bebas.")

st.divider()

# --- BAGIAN 3: VISUALISASI ---
st.header("3. Visualisasi Eksperimen")

# Terapkan gaya bawaan matplotlib seperti di Colab
plt.style.use('seaborn-v0_8-darkgrid')

col3, col4 = st.columns(2)

with col3:
    # Visualisasi 1: Heatmap
    st.subheader("1. Heatmap Korelasi")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".3f", linewidths=.5, annot_kws={"size": 10}, ax=ax1)
    st.pyplot(fig1)

    # Visualisasi 3: Residual Plot
    st.subheader("3. Uji Heteroskedastisitas (Residual Plot)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(y_pred, residuals, alpha=0.5, color='#e74c3c', edgecolors='k')
    ax3.axhline(y=0, color='#2c3e50', linestyle='--', linewidth=2)
    ax3.set_xlabel('Nilai Prediksi')
    ax3.set_ylabel('Residual')
    ax3.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig3)

with col4:
    # Visualisasi 2: Scatter Aktual vs Prediksi
    st.subheader("2. Scatter Plot: Aktual vs Prediksi")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_test, y_pred, alpha=0.6, color='#3498db', label='Data Prediksi', edgecolors='k')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#e74c3c', linestyle='--', lw=2, label='Garis Ideal')
    ax2.set_xlabel('Nilai Aktual (Close)')
    ax2.set_ylabel('Nilai Prediksi (Close)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig2)

    # Visualisasi 4: Distribusi Residual
    st.subheader("4. Uji Normalitas (Distribusi Residual)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color='#3498db', bins=30, edgecolor='black', alpha=0.7, ax=ax4)
    ax4.set_xlabel('Residual')
    ax4.set_ylabel('Frekuensi')
    ax4.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig4)

st.divider()

# --- BAGIAN 4: SIMULASI & PREDIKSI (WHAT-IF ANALYSIS) ---
st.header("4. Simulasi dan Prediksi Harga (What-If Analysis)")
st.markdown("Uji model regresi dengan memasukkan nilai skenario **Open**, **High**, **Low**, dan **Volume** Anda sendiri untuk melihat estimasi harga penutupan (**Close**) secara *real-time*.")

# Ambil data hari terakhir sebagai nilai default agar realistis
latest_data = df.iloc[-1]

col_in1, col_in2, col_in3, col_in4 = st.columns(4)

with col_in1:
    input_open = st.number_input(
        "Harga Open ($)", 
        min_value=0.0, 
        value=float(latest_data['Open']), 
        step=100.0, 
        format="%.2f",
        help="Masukkan asumsi harga pembukaan"
    )
with col_in2:
    input_high = st.number_input(
        "Harga High ($)", 
        min_value=0.0, 
        value=float(latest_data['High']), 
        step=100.0, 
        format="%.2f",
        help="Masukkan asumsi harga tertinggi"
    )
with col_in3:
    input_low = st.number_input(
        "Harga Low ($)", 
        min_value=0.0, 
        value=float(latest_data['Low']), 
        step=100.0, 
        format="%.2f",
        help="Masukkan asumsi harga terendah"
    )
with col_in4:
    input_volume = st.number_input(
        "Volume Transaksi", 
        min_value=0.0, 
        value=float(latest_data['Volume']), 
        step=1000000.0, 
        format="%.0f",
        help="Masukkan asumsi volume perdagangan harian"
    )

# Siapkan DataFrame baru sesuai format yang diminta statsmodels (menambahkan kolom konstanta)
input_df = pd.DataFrame({
    'const': [1.0],
    'Open': [input_open],
    'High': [input_high],
    'Low': [input_low],
    'Volume': [input_volume]
})

# Eksekusi prediksi dengan model OLS yang telah dilatih
pred_close = model.predict(input_df)[0]
selisih_open_close = pred_close - input_open

# Tampilkan hasil prediksi
st.subheader("💡 Hasil Estimasi Model")
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    st.metric(
        label="Prediksi Harga Close", 
        value=f"${pred_close:,.2f}", 
        delta=f"{selisih_open_close:,.2f} dari harga Open"
    )

with col_res2:
    # Memberikan interpretasi otomatis berdasarkan hasil hitungan
    arah_tren = "mengalami KENAIKAN" if selisih_open_close >= 0 else "mengalami PENURUNAN"
    st.info(
        f"**Analisis Skenario:**\n\n"
        f"Berdasarkan parameter yang dimasukkan, model memprediksi harga Bitcoin akan **{arah_tren}** sebesar "
        f"**${abs(selisih_open_close):,.2f}** dari harga pembukaan (Open). "
        f"Perlu diingat bahwa pada regresi linier harga Bitcoin, variabel **High** dan **Low** umumnya memiliki bobot koefisien "
        f"paling dominan dalam menarik arah prediksi akhir."
    )
