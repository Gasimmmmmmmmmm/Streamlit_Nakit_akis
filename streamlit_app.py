import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# --- Yapılandırma ---
CONFIG = {
    "MODEL_PATH": "/content/nakit_0_07%_modeli.h5",
    "SEQ_LENGTH": 10,
    "PREDICTION_DAYS": 30,
    "CURRENCY_SYMBOL": "₺",
    "DATE_COLUMN": "tarih",
    "CASH_IN_KEYWORDS": ["Gelir", "gelir", "Giriş", "giriş", "Giris", "giris"],
    "CASH_OUT_KEYWORDS": ["Gider", "gider", "Çıkış", "çıkış", "Cikis", "cikis"]
}

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="💰 Nakit Akışı AI Tahmin Sistemi",
    layout="wide",
    page_icon="💹"
)

# --- Sidebar: Bilgi ve Durum Paneli ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965507.png", width=60)
    st.title("⚙️ Sistem Bilgisi")
    st.markdown("---")

# --- Başlık ve Açıklama ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🤖 LSTM Destekli Nakit Akışı Tahmin Sistemi</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-bottom: 2rem;'>
    Nakit akışınızı analiz edin, geleceği öngörün ve finansal kararlarınızı veriyle destekleyin.
</div>
""", unsafe_allow_html=True)

# --- Model Yükleme ---
@st.cache_resource
def load_lstm_model(model_path):
    try:
        model = load_model(model_path)
        return model, True, None
    except Exception as e:
        return None, False, str(e)

model, model_loaded, model_error = load_lstm_model(CONFIG["MODEL_PATH"])

with st.sidebar:
    if model_loaded:
        st.success("✅ LSTM modeli yüklendi")
    else:
        st.error("❌ Model yüklenemedi")
        st.code(model_error)
        st.stop()

# --- Veri Yükleme ---
uploaded_file = st.file_uploader("📂 Nakit akışı verinizi (CSV) yükleyin", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("📤 Lütfen bir CSV dosyası yükleyin. Dosya sütunlarında tarih, gelir ve gider bilgileri olmalıdır.", icon="ℹ️")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Dosya okuma hatası: {e}")
    st.stop()

if df.empty:
    st.error("❌ Yüklenen dosya boş.")
    st.stop()

# --- 🧩 Sütun İsmi Düzeltme ---
df.columns = [str(col).strip() for col in df.columns]
lowercase_cols = {col.lower(): col for col in df.columns}
expected_col = CONFIG["DATE_COLUMN"].lower()

if expected_col not in lowercase_cols:
    possible_match = None
    for col in df.columns:
        if col.lower() == expected_col or expected_col in col.lower():
            possible_match = col
            break
    if possible_match:
        df.rename(columns={possible_match: CONFIG["DATE_COLUMN"]}, inplace=True)
        st.warning(f"⚠️ '{possible_match}' sütunu '{CONFIG['DATE_COLUMN']}' olarak yeniden adlandırıldı.")
    else:
        st.error(f"❌ '{CONFIG['DATE_COLUMN']}' sütunu bulunamadı. Lütfen verinizde tarih bilgisi olduğundan emin olun.")
        st.stop()
else:
    true_name = lowercase_cols[expected_col]
    if true_name != CONFIG["DATE_COLUMN"]:
        df.rename(columns={true_name: CONFIG["DATE_COLUMN"]}, inplace=True)
        st.info(f"ℹ️ '{true_name}' sütunu '{CONFIG['DATE_COLUMN']}' olarak düzeltildi.")

with st.sidebar:
    st.success(f"✅ {len(df)} kayıt yüklendi")

# --- Tarih İşleme ---
date_col = CONFIG["DATE_COLUMN"]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

# --- Gelir ve Gider sütunlarını otomatik bul ---
def find_columns(df, keywords):
    return [col for col in df.columns if any(kw in col for kw in keywords)]

cash_in_cols = find_columns(df, CONFIG["CASH_IN_KEYWORDS"])
cash_out_cols = find_columns(df, CONFIG["CASH_OUT_KEYWORDS"])

if not cash_in_cols:
    st.warning("⚠️ Gelir sütunu otomatik bulunamadı. 'Gelir', 'Giriş' gibi anahtar kelimeler içermelidir.")
if not cash_out_cols:
    st.warning("⚠️ Gider sütunu otomatik bulunamadı. 'Gider', 'Çıkış' gibi anahtar kelimeler içermelidir.")

if not cash_in_cols or not cash_out_cols:
    st.stop()

# --- 🔧 Otomatik Temizleme ve Sayıya Dönüştürme ---
def clean_numeric_column(series):
    return (
        series.astype(str)
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(",", ".", regex=True)
        .replace("", np.nan)
        .astype(float)
        .fillna(0.0)
    )

for col in cash_in_cols + cash_out_cols:
    df[col] = clean_numeric_column(df[col])

# --- Hesaplamalar ---
df["Toplam Gelir"] = df[cash_in_cols].sum(axis=1)
df["Toplam Gider"] = df[cash_out_cols].sum(axis=1)
df["Net Nakit"] = df["Toplam Gelir"] - df["Toplam Gider"]
df["Birikimli Bakiye"] = df["Net Nakit"].cumsum()

# --- Özet Metrikler ---
st.markdown("### 📊 Finansal Özet")
col1, col2, col3, col4 = st.columns(4)
total_income = df['Toplam Gelir'].sum()
total_expense = df['Toplam Gider'].sum()
final_balance = df['Birikimli Bakiye'].iloc[-1]
avg_daily_net = df['Net Nakit'].mean()

col1.metric("Toplam Gelir", f"{total_income:,.0f} {CONFIG['CURRENCY_SYMBOL']}")
col2.metric("Toplam Gider", f"{total_expense:,.0f} {CONFIG['CURRENCY_SYMBOL']}", delta_color="inverse")
col3.metric("Son Bakiye", f"{final_balance:,.0f} {CONFIG['CURRENCY_SYMBOL']}")
col4.metric("Günlük Ort. Net", f"{avg_daily_net:,.0f} {CONFIG['CURRENCY_SYMBOL']}")

# --- Sekmeler ---
tab1, tab2, tab3 = st.tabs(["📈 Analiz ve Trendler", "🔮 AI Tahmin Sonuçları", "📋 Ham Veri"])

# --- Tab 1 ---
with tab1:
    st.subheader("Gelir, Gider ve Bakiye Zaman Serisi")
    fig1 = px.line(df, x=date_col, y=["Toplam Gelir", "Toplam Gider"],
                   title="Gelir & Gider Trendleri",
                   labels={"value": "Tutar (₺)", "tarih": "Tarih"},
                   color_discrete_sequence=["#2ecc71", "#e74c3c"])
    fig1.update_traces(mode="lines+markers", marker=dict(size=4))
    fig1.update_layout(hovermode="x unified", title_x=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.area(df, x=date_col, y="Birikimli Bakiye",
                   title="Birikimli Nakit Bakiyesi",
                   labels={"Birikimli Bakiye": "Bakiye (₺)", "tarih": "Tarih"},
                   color_discrete_sequence=["#3498db"])
    fig2.update_layout(title_x=0.5)
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2 ---
with tab2:
    st.subheader(f"🧠 {CONFIG['PREDICTION_DAYS']} Günlük Nakit Akışı Tahmini (LSTM)")
    scaler = MinMaxScaler()
    net_cash = df[["Net Nakit"]].values
    scaled_data = scaler.fit_transform(net_cash)
    seq_len = CONFIG["SEQ_LENGTH"]

    if len(scaled_data) <= seq_len:
        st.error("⚠️ Tahmin için yeterli veri yok. En az 11 günlük veri gereklidir.")
        st.stop()

    X = [scaled_data[i:i + seq_len] for i in range(len(scaled_data) - seq_len)]
    X = np.array(X)

    last_seq = X[-1].reshape(1, seq_len, 1)
    preds = []
    current = last_seq.copy()

    for _ in range(CONFIG["PREDICTION_DAYS"]):
        pred = model.predict(current, verbose=0)
        preds.append(pred[0, 0])
        current = np.append(current[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = [df[date_col].max() + timedelta(days=i + 1) for i in range(CONFIG["PREDICTION_DAYS"])]

    tahmin_df = pd.DataFrame({"tarih": future_dates, "Tahmini Net Nakit": preds})
    tahmin_df["Kümülatif Tahmini Bakiye"] = tahmin_df["Tahmini Net Nakit"].cumsum() + final_balance

    fig3 = px.line(tahmin_df, x="tarih", y="Tahmini Net Nakit",
                   title="Günlük Tahmini Net Nakit Akışı",
                   labels={"Tahmini Net Nakit": "Net Nakit (₺)", "tarih": "Tarih"},
                   markers=True,
                   color_discrete_sequence=["#9b59b6"])
    fig3.update_traces(mode="lines+markers", marker=dict(size=5))
    fig3.update_layout(title_x=0.5, hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(tahmin_df, x="tarih", y="Kümülatif Tahmini Bakiye",
                   title="Tahmini Birikimli Bakiye",
                   labels={"Kümülatif Tahmini Bakiye": "Bakiye (₺)", "tarih": "Tarih"},
                   color_discrete_sequence=["#f39c12"])
    fig4.update_layout(title_x=0.5)
    st.plotly_chart(fig4, use_container_width=True)

    avg_pred = tahmin_df["Tahmini Net Nakit"].mean()
    if avg_pred < 0:
        st.error("🔴 **Uyarı:** Ortalama günlük nakit akışı negatif. Nakit açığı riski var.")
    elif avg_pred > 0:
        st.success("🟢 **İyi haber:** Ortalama günlük nakit akışı pozitif.")
    else:
        st.warning("🟡 Nakit akışı nötr seviyede. Dikkatli izleme önerilir.")

    st.download_button(
        "📥 Tahmin Sonuçlarını İndir (CSV)",
        tahmin_df.to_csv(index=False).encode("utf-8"),
        file_name="nakit_akisi_tahmini.csv",
        mime="text/csv",
        use_container_width=True
    )

# --- Tab 3 ---
with tab3:
    st.subheader("Veri Önizleme (Son 50 Kayıt)")
    st.dataframe(df.tail(50), use_container_width=True)
    with st.expander("🔍 Veri Yapılandırması"):
        st.write(f"- **Tarih sütunu:** `{date_col}`")
        st.write(f"- **Gelir sütunları:** {cash_in_cols}")
        st.write(f"- **Gider sütunları:** {cash_out_cols}")
        st.write(f"- **Toplam kayıt:** {len(df)}")
