import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. KONFIGURASI & CUSTOM CSS (SAAS LOOK) ---
st.set_page_config(page_title="BTC Analytics Hub", page_icon="", layout="wide")

# Custom CSS untuk tampilan kartu, font, dan menghilangkan elemen bawaan Streamlit yang kaku
st.markdown("""
<style>
    /* Styling umum & container utama */
    .main {
        background-color: #F8FAFC;
    }
    /* Kustomisasi Kartu Metric */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #CBD5E1;
    }
    /* Headers typography */
    h1, h2, h3 {
        color: #0F172A;
        font-family: 'Inter', sans-serif;
    }
    /* Custom info box */
    .custom-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #1E3A8A;
    }
    /* Sembunyikan menu default dan footer agar bersih seperti aplikasi web asli */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD & PREPARE DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_historis_btc.csv', sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()

# Preparation Model OLS
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()

X_test_sm = sm.add_constant(X_test)
y_pred = model.predict(X_test_sm)
residuals = y_test - y_pred

# --- 3. HEADER & KPI OVERVIEW ---
st.title(""Dashboard Prediksi Bitcoins")
st.markdown("<p style='color: #64748B; font-size: 1.1rem; margin-top: -10px; margin-bottom: 24px;'>Platform analisis regresi historis dan simulasi skenario harga Bitcoin secara real-time.</p>", unsafe_allow_html=True)

# Top KPI Section (Langsung memberi konteks ke pengguna tanpa tabel mentah yang membosankan)
latest_data = df.iloc[-1]
prev_data = df.iloc[-2]
price_change = latest_data['Close'] - prev_data['Close']
pct_change = (price_change / prev_data['Close']) * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Harga Penutupan Terkini", f"${latest_data['Close']:,.2f}", f"{price_change:,.2f} ({pct_change:.2f}%)")
with kpi2:
    st.metric("Rentang Periode", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}", f"{len(df):,} Hari Observasi", delta_color="off")
with kpi3:
    st.metric("Akurasi Model (R²)", f"{model.rsquared * 100:.2f}%", "Sangat Kuat", delta_color="normal")
with kpi4:
    st.metric("Rata-rata Volume", f"{df['Volume'].mean()/1e9:.2f}B", "USD / Hari", delta_color="off")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. TABS STRUKTUR YANG BERSIH ---
tab_simulasi, tab_visual, tab_diagnostik = st.tabs([
    "Simulator Skenario", 
    "Analisis Visual & Korelasi", 
    "Diagnostik Statistik (OLS)"
])

# ==========================================
# TAB 1: SIMULATOR (WHAT-IF ANALYSIS)
# ==========================================
with tab_simulasi:
    st.markdown("""
    <div class="custom-box">
        <strong>Ruang Simulasi:</strong> Sesuaikan parameter pasar di bawah ini untuk melihat estimasi penutupan harga (Close) berdasarkan bobot regresi linier historis.
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("#### Parameter Input Pasar")
        with st.container(border=True):
            input_open = st.number_input("Harga Pembukaan (Open USD)", min_value=0.0, value=float(latest_data['Open']), step=50.0, format="%.2f")
            input_high = st.number_input("Target Tertinggi (High USD)", min_value=0.0, value=float(latest_data['High']), step=50.0, format="%.2f")
            input_low = st.number_input("Batas Terendah (Low USD)", min_value=0.0, value=float(latest_data['Low']), step=50.0, format="%.2f")
            input_volume = st.number_input("Volume Perdagangan (USD)", min_value=0.0, value=float(latest_data['Volume']), step=1000000.0, format="%.0f")
            
            # Kalkulasi langsung
            input_df = pd.DataFrame({'const': [1.0], 'Open': [input_open], 'High': [input_high], 'Low': [input_low], 'Volume': [input_volume]})
            pred_close = model.predict(input_df)[0]
            delta_open = pred_close - input_open
    
    with col_output:
        st.markdown("#### Proyeksi Hasil")
        with st.container(border=True):
            st.write("Estimasi Harga Penutupan (Close)")
            st.markdown(f"<h1 style='font-size: 3rem; color: #0F172A; margin: 0;'>${pred_close:,.2f}</h1>", unsafe_allow_html=True)
            
            # Badge delta custom
            color_delta = "#10B981" if delta_open >= 0 else "#EF4444"
            sign_delta = "+" if delta_open >= 0 else ""
            st.markdown(f"<p style='color: {color_delta}; font-weight: 600; font-size: 1.1rem;'>{sign_delta}${delta_open:,.2f} dari harga Open</p>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown("**Catatan Analis:**")
            if delta_open >= 0:
                st.write("Skenario parameter ini mengindikasikan tekanan beli yang dominan, mendorong harga penutupan lebih tinggi dari pembukaan.")
            else:
                st.write("Skenario ini memperlihatkan koreksi harian, di mana batas terendah (*Low*) menarik rata-rata penutupan ke zona merah.")
            
            st.caption("Prediksi dihitung secara instan menggunakan bobot koefisien OLS tanpa intervensi manual.")

# ==========================================
# TAB 2: VISUALISASI DATA (FINTECH STYLE)
# ==========================================
with tab_visual:
    # Set styling matplotlib agar bersih minimalis (tanpa kotak hitam atau background abu-abu Google Colab)
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = 'Inter, Arial, sans-serif'
    plt.rcParams['axes.edgecolor'] = '#E2E8F0'
    plt.rcParams['axes.linewidth'] = 0.8
    
    col_v1, col_v2 = st.columns(2, gap="medium")
    
    with col_v1:
        st.markdown("#### Matriks Korelasi Pasar")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        # Menggunakan palette 'Blues' bergaya minimalis
        sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".3f", cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 10, "weight": "bold"}, ax=ax1)
        ax1.tick_params(colors='#475569', size=0)
        st.pyplot(fig1)
        
        st.markdown("#### Uji Homoskedastisitas (Residual vs Prediksi)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(y_pred, residuals, alpha=0.6, color='#3B82F6', edgecolors='none', s=30)
        ax3.axhline(y=0, color='#EF4444', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('Nilai Prediksi USD', color='#64748B', fontsize=9)
        ax3.set_ylabel('Residual', color='#64748B', fontsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
        ax3.tick_params(colors='#475569', labelsize=8)
        st.pyplot(fig3)

    with col_v2:
        st.markdown("#### Presisi Model: Aktual vs Prediksi")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.scatter(y_test, y_pred, alpha=0.5, color='#0EA5E9', label='Data Uji', edgecolors='none', s=25)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#0F172A', linestyle='-', lw=1.5, label='Garis Presisi 100%')
        ax2.set_xlabel('Aktual (Close USD)', color='#64748B', fontsize=9)
        ax2.set_ylabel('Prediksi (Close USD)', color='#64748B', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
        ax2.tick_params(colors='#475569', labelsize=8)
        ax2.legend(frameon=False, loc='upper left')
        st.pyplot(fig2)
        
        st.markdown("#### Distribusi Normalitas Residual")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, color='#6366F1', bins=35, edgecolor='white', alpha=0.8, ax=ax4)
        ax4.set_xlabel('Error / Residual', color='#64748B', fontsize=9)
        ax4.set_ylabel('Frekuensi', color='#64748B', fontsize=9)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
        ax4.tick_params(colors='#475569', labelsize=8)
        st.pyplot(fig4)

# ==========================================
# TAB 3: DIAGNOSTIK STATISTIK
# ==========================================
with tab_diagnostik:
    st.markdown("#### Ringkasan Parameter OLS & Multikolinearitas")
    col_d1, col_d2 = st.columns([1.5, 1], gap="large")
    
    with col_d1:
        st.write("**Model OLS Regression Output**")
        # Menampilkan resume dengan code block agar monospaced dan rapi untuk auditor statistik
        st.code(model.summary().as_text(), language="text")
        
    with col_d2:
        st.write("**Variance Inflation Factor (VIF)**")
        vif_data = pd.DataFrame()
        vif_data["Fitur / Variabel"] = X.columns
        vif_data["Skor VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        
        st.dataframe(
            vif_data.style.format({"Skor VIF": "{:.2f}"}), 
            use_container_width=True,
            hide_index=True
        )
        st.caption("**Rule of Thumb:** Pada data *time-series* finansial harian (OHLC), skor VIF antar harga (Open, High, Low) umumnya sangat tinggi karena pergerakannya secara alami saling mengikuti (*highly co-integrated*).")
        
        st.divider()
        st.write("**Data Historis Mentah (Sample)**")
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)
