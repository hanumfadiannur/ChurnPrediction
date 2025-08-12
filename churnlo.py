import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jl
from sklearn.ensemble import GradientBoostingClassifier

# ---------- Setup ----------
st.set_page_config(page_title="Prediksi Pelanggan", layout="wide")
st.title("📊 Prediksi Pelanggan Potensial Churn")
st.markdown("Masukkan data pelanggan untuk memprediksi apakah mereka berpotensi berdasarkan model machine learning.")

# ---------- Load Model dan Artifacts ----------
model = jl.load("model4.joblib")
encoders = jl.load("encoders.jblib")  
scaler = jl.load("scaler.jblib")

# ---------- Sidebar Input ----------
def get_user_input():
    st.sidebar.header("📝 Input Pelanggan")
    bundling = st.sidebar.selectbox("Bundling", ["1P", "2P", "3P"])
    status_paid = st.sidebar.selectbox("Status Paid", ["PAID", "UNPAID"])
    product = st.sidebar.selectbox("Product", ["TLP", "TDSL"])
    status_billper = st.sidebar.selectbox("Status Billper", ["EXISTING", "BILLPER"])
    umur = st.sidebar.selectbox("Umur", ["0-3 Bulan", "4-6 Bulan", "7-9 Bulan", "10-12 Bulan", "1-2 Tahun", ">2 Tahun"])
    status_tagihan = st.sidebar.selectbox("Status Tagihan", ["TAGIHAN NAIK", "TAGIHAN BARU", "TAGIHAN SAMA", "TAGIHAN TURUN"])
    habit = st.sidebar.selectbox("Habit", ["HABIT JELEK", "UMUR <= 6 Bulan", "HABIT BAGUS"])
    keterangan = st.sidebar.selectbox("Keterangan", [
        "CT0", "MUTASI", "DEBIT-NAIK", "ABON_NOL", "DEBIT-TURUN", "KREDIT-TURUN",
        "BILING-2-TURUN", "USAGE-TURUN", "ABON-TURUN", "BILING-2-NAIK", "KREDIT-NAIK",
        "TAGIHAN SAMA", "USAGE-NAIK", "ABON-NAIK", "PSB"
    ])
    status_saldo = st.sidebar.selectbox("Status Saldo", ["A", "B", "C", "D", "E", "F", "G", "P"])
    total_tagihan = st.sidebar.number_input("Total Tagihan", min_value=0, max_value=10_000_000_000, value=500_000, step=1)

    return {
        "bundling": bundling,
        "status_paid": status_paid,
        "product": product,
        "status_billper": status_billper,
        "umur": umur,
        "status_tagihan": status_tagihan,
        "habit": habit,
        "keterangan": keterangan,
        "status_saldo": status_saldo,
        "total_tagihan": total_tagihan,
    }

# ---------- Preprocessing ----------
def preprocess(raw):
    enc = encoders
    x = {
        "BUNDLING": enc["bundling_map"].get(raw["bundling"], np.nan),
        "STATUS_PAID": enc["status_paid_map"].get(raw["status_paid"], np.nan),
        "PRODUCT": enc["product_map"].get(raw["product"], np.nan),
        "STATUS_BILLPER": enc["status_billper_map"].get(raw["status_billper"], np.nan),
        "UMUR": enc["umur_map"].get(raw["umur"], np.nan),
        "STATUS_TAGIHAN": enc["status_tagihan_map"].get(raw["status_tagihan"], np.nan),
        "HABIT": enc["habit_map"].get(raw["habit"], np.nan),
        "KETERANGAN": enc["keterangan_map"].get(raw["keterangan"], np.nan),
        "TOTAG": raw["total_tagihan"],
        "STATUS_SALDO": enc["saldo_map"].get(raw["status_saldo"], np.nan),
    }
    df = pd.DataFrame([x])
    df[['TOTAG']] = scaler.transform(df[['TOTAG']])
    return df


# ---------- Probabilitas Visual ----------
def show_probability_chart(prob):
    prob_df = pd.DataFrame({
        "Kelas": ["Tidak Potensial", "Potensial"],
        "Probabilitas": prob
    })
    fig, ax = plt.subplots()
    sns.barplot(data=prob_df, x="Probabilitas", y="Kelas", palette="coolwarm", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Confidence Score")
    st.pyplot(fig)

# ---------- Jalankan ----------
user_input = get_user_input()
processed = preprocess(user_input)
prediction = model.predict(processed)[0]
label_map = {0: "Tidak Potensial", 1: "Potensial Churn"}

# ---------- Tampilkan Hasil ----------
st.subheader("🧾 Hasil Prediksi")
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(processed)[0]
    if prediction == 1:
        st.success(f"✅ Pelanggan diprediksi **Potensial Churn** dengan kepercayaan {100 * prob[1]:.2f}%")
    else:
        st.error(f"⚠️ Pelanggan diprediksi **Tidak Potensial** dengan kepercayaan {100 * prob[0]:.2f}%")
    show_probability_chart(prob)
else:
    st.write(f"Prediksi: **{label_map.get(prediction)}**")


# ---------- Batch Prediction via Upload ----------
st.markdown("---")
st.header("📁 Batch Prediksi dari File")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], help="Format file harus CSV atau Excel (XLSX/XLS).")


def preprocess_df(df_raw):
    data_list = []
    for _, raw in df_raw.iterrows():
        row = {
            "BUNDLING": encoders["bundling_map"].get(raw["BUNDLING"], np.nan),
            "STATUS_PAID": encoders["status_paid_map"].get(raw["STATUS_PAID"], np.nan),
            "PRODUCT": encoders["product_map"].get(raw["PRODUCT"], np.nan),
            "STATUS_BILLPER": encoders["status_billper_map"].get(raw["STATUS_BILLPER"], np.nan),
            "UMUR": encoders["umur_map"].get(raw["UMUR"], np.nan),
            "STATUS_TAGIHAN": encoders["status_tagihan_map"].get(raw["STATUS_TAGIHAN"], np.nan),
            "HABIT": encoders["habit_map"].get(str(raw["HABIT"]).upper(), np.nan),
            "KETERANGAN": encoders["keterangan_map"].get(raw["KETERANGAN"], np.nan),
            "STATUS_SALDO": encoders["saldo_map"].get(raw["STATUS_SALDO"], np.nan),
            "TOTAG": raw["TOTAG"],
        }
        data_list.append(row)

    FEATURE_ORDER = [
        "BUNDLING",
        "STATUS_PAID",
        "PRODUCT",
        "STATUS_BILLPER",
        "UMUR",
        "STATUS_TAGIHAN",
        "HABIT",
        "KETERANGAN",
        "TOTAG",
        "STATUS_SALDO"
    ]

    def clean_money_column(value):
        try:
            return float(str(value).replace(".", "").replace(",", ".").strip())
        except:
            return np.nan

    # Buat DataFrame & urutkan sesuai fitur
    df = pd.DataFrame(data_list)[FEATURE_ORDER]

    # Bersihkan nilai uang
    df["TOTAG"] = df["TOTAG"].apply(clean_money_column)

    # --- Handle Missing Values ---
    # Untuk kolom kategori: isi NaN dengan 0 (asumsi 0 adalah kode "tidak diketahui")
    cat_cols = ["BUNDLING", "STATUS_PAID", "PRODUCT", "STATUS_BILLPER", "UMUR",
                "STATUS_TAGIHAN", "HABIT", "KETERANGAN", "STATUS_SALDO"]
    df[cat_cols] = df[cat_cols].fillna(0)

    # Untuk kolom numerik: isi NaN dengan median (bisa juga 0 kalau mau)
    df["TOTAG"] = df["TOTAG"].fillna(df["TOTAG"].median())

    # Scaling
    df[["TOTAG"]] = scaler.transform(df[["TOTAG"]])

    return df
import os

if uploaded_file is not None:
    try:
        # Deteksi jenis file dari ekstensi
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

        if file_ext in [".xlsx", ".xls"]:
            df_csv = pd.read_excel(uploaded_file, dtype=str)
        elif file_ext == ".csv":
            df_csv = pd.read_csv(uploaded_file, encoding="latin1", dtype=str)
        else:
            st.error("❌ Format file tidak didukung. Gunakan CSV atau Excel.")
            st.stop()

        # Normalisasi nama kolom
        df_csv.columns = df_csv.columns.str.upper().str.replace(" ", "_")

        required_cols = [
            "BUNDLING", "STATUS_PAID", "PRODUCT", "STATUS_BILLPER", "UMUR",
            "STATUS_TAGIHAN", "HABIT", "KETERANGAN", "STATUS_SALDO", "TOTAG"
        ]

        if all(col in df_csv.columns for col in required_cols):
            st.success("✅ File berhasil dibaca. Menjalankan prediksi...")

            st.subheader("📋 Data yang Diupload")
            st.dataframe(df_csv)

            processed_batch = preprocess_df(df_csv)
            pred_batch = model.predict(processed_batch)

            df_csv["Prediksi"] = ["Potensial Churn" if p == 1 else "Tidak Potensial" for p in pred_batch]

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(df_csv)

            st.subheader("📈 Distribusi Hasil Prediksi")
            pred_counts = df_csv["Prediksi"].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%", startangle=90, colors=["#ff9999", "#66b3ff"])
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

            csv_export = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Hasil ke CSV",
                data=csv_export,
                file_name="hasil_prediksi_batch.csv",
                mime="text/csv"
            )

        else:
            st.error(f"❌ Kolom tidak lengkap. Kolom wajib: {', '.join(required_cols)}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
