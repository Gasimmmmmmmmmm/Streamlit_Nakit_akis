import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import re
import unicodedata

# --- Yapılandırma ---
CONFIG = {
    "MODEL_PATH": "nakit_0_07%_modeli.h5",
    "SEQ_LENGTH": 10,
    "PREDICTION_DAYS": 30,
    "CURRENCY_SYMBOL": "₺",
    "CASH_IN_KEYWORDS": ["Gelir", "gelir", "Giriş", "giriş", "Giris", "giris"],
    "CASH_OUT_KEYWORDS": ["Gider", "gider", "Çıkış", "çıkış", "Cikis", "cikis"]
}

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="💰 Nakit Akışı AI Tahmin Sistemi",
    layout="wide",
    page_icon="💹"
)

# --- Sidebar: Başlık ---
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

# --- 🧩 Otomatik Sütun Adı Düzeltme ---
original_columns = df.columns.tolist()

def normalize_column_name(name):
    name = str(name).strip()
    name = " ".join(name.split())  # fazla boşlukları temizle
    turkish_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    name = name.translate(turkish_map)
    name = re.sub(r"\s+", "_", name)  # boşlukları alt çizgi yap
    name = unicodedata.normalize("NFKD", name)
    name = "_".join(word.capitalize() for word in name.split("_"))
    return name

df.columns = [normalize_column_name(col) for col in df.columns]

if original_columns != df.columns.tolist():
    st.info("🔤 Sütun adları otomatik olarak düzenlendi ve normalize edildi.", icon="ℹ️")
    diffs = [f"**{o} → {n}**" for o, n in zip(original_columns, df.columns) if o != n]
    if diffs:
        st.markdown("Düzenlenen sütunlar: " + ", ".join(diffs))

# --- 🔧 VERİ YAPILANDIRMA ---
with st.sidebar:
    st.markdown("## ⚙️ Veri Yapılandırma")
    st.caption("Yüklenen veri setinden tarih, gelir ve gider sütunlarını seçin.")

    # --- 1️⃣ Sütun Tipi Düzenleme ---
    st.markdown("### 🧩 Sütun Tipi Düzenleme")
    st.caption("Her sütunun veri tipini kontrol edin veya değiştirin.")

    type_options = ["string", "float", "int", "datetime", "bool"]
    column_type_map = {}

    for col in df.columns:
        detected_type = str(df[col].dtype)
        if "date" in col.lower() or "tarih" in col.lower():
            suggested_type = "datetime"
        elif np.issubdtype(df[col].dtype, np.number):
            suggested_type = "float"
        elif detected_type == "bool":
            suggested_type = "bool"
        else:
            suggested_type = "string"

        selected_type = st.selectbox(
            f"🧱 {col} sütunu tipi:",
            type_options,
            index=type_options.index(suggested_type),
            key=f"type_select_{col}"
        )

        sample_values = df[col].dropna().astype(str).head(3).tolist()
        st.caption(f"Örnek değerler: {sample_values}")
        column_type_map[col] = selected_type

    st.markdown("---")

    # --- 2️⃣ Tarih / Gelir / Gider sütunları seçimi ---
    date_col = st.selectbox("📅 Tarih sütunu", df.columns, index=0)
    cash_in_cols = st.multiselect("🟢 Gelir sütunları", df.columns,
                                  default=[col for col in df.columns if any(kw.lower() in col.lower() for kw in CONFIG["CASH_IN_KEYWORDS"])])
    cash_out_cols = st.multiselect("🔴 Gider sütunları", df.columns,
                                   default=[col for col in df.columns if any(kw.lower() in col.lower() for kw in CONFIG["CASH_OUT_KEYWORDS"])])

    confirm = st.button("✅ Yapılandırmayı Onayla")

if not confirm:
    st.warning("Lütfen sütun tiplerini ve veri sütunlarını yapılandırın, ardından 'Yapılandırmayı Onayla' butonuna basın.", icon="⚠️")
    st.stop()

st.success("✅ Veri yapılandırması tamamlandı ve sütun tipleri uygulanıyor...")

# --- Tip Dönüşümleri ---
for col, dtype in column_type_map.items():
    try:
        if dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        elif dtype == "bool":
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(str)
    except Exception as e:
        st.warning(f"⚠️ `{col}` sütunu {dtype} tipine dönüştürülemedi: {e}")

original_dtypes = df.dtypes.to_dict()

# --- Model Yükleme ---
model, model_loaded, model_error = load_lstm_model(CONFIG["MODEL_PATH"])
with st.sidebar:
    if model_loaded:
        st.success("✅ LSTM modeli yüklendi")
    else:
        st.error("❌ Model yüklenemedi")
        st.code(model_error)
        st.stop()

# --- Veri Temizleme ---
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

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

# Hesaplanan sütunlar
df["Toplam Gelir"] = df[cash_in_cols].sum(axis=1)
df["Toplam Gider"] = df[cash_out_cols].sum(axis=1)
df["Net Nakit"] = df["Toplam Gelir"] - df["Toplam Gider"]
df["Birikimli Bakiye"] = df["Net Nakit"].cumsum()

# --- Özet ---
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
                   labels={"value": "Tutar (₺)", date_col: "Tarih"},
                   color_discrete_sequence=["#2ecc71", "#e74c3c"])
    fig1.update_traces(mode="lines+markers", marker=dict(size=4))
    fig1.update_layout(hovermode="x unified", title_x=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.area(df, x=date_col, y="Birikimli Bakiye",
                   title="Birikimli Nakit Bakiyesi",
                   labels={"Birikimli Bakiye": "Bakiye (₺)", date_col: "Tarih"},
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
