import streamlit as st
import pandas as pd
import pickle

# Load model
with open("fraud_model.pkl", "rb") as file:
    model, columns = pickle.load(file)

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Financial Fraud Detection System")
st.write("AI-based Financial Fraud Detection")

st.divider()

st.subheader("📁 Upload Transaction File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV only once
        original_df = pd.read_csv(uploaded_file)

        if original_df.empty:
            st.error("❌ The uploaded CSV file is empty.")
            st.stop()

        st.success("✅ CSV uploaded successfully!")

        st.subheader("📋 Uploaded Data")
        st.dataframe(
            original_df.head(100),
            use_container_width=True
        )

        # Copy for prediction
        df = original_df.copy()

        # Remove target column
        for col in ["isFraud", "is_fraud"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Remove ID columns
        for col in ["id", "ID", "transaction_id", "TransactionID"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Convert categorical columns
        df = pd.get_dummies(df)

        # Match training columns
        df = df.reindex(
            columns=columns,
            fill_value=0
        )

        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        # Create result from original data
        result = original_df.copy()

        result["Fraud Prediction"] = predictions
        result["Fraud Probability"] = (
            probabilities * 100
        ).round(2)

        # Statistics
        total = len(result)
        fraud = int((predictions == 1).sum())
        genuine = total - fraud

        fraud_rate = (
            fraud / total * 100
            if total > 0 else 0
        )

        st.divider()

        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            total
        )

        col2.metric(
            "🚨 Fraud",
            fraud
        )

        col3.metric(
            "✅ Genuine",
            genuine
        )

        col4.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        st.divider()

        # Results
        st.subheader("🔍 Prediction Results")

        st.dataframe(
            result,
            use_container_width=True
        )

        # Chart
        st.subheader("📊 Fraud Analysis")

        chart_data = pd.DataFrame({
            "Transaction Type": [
                "Genuine",
                "Fraud"
            ],
            "Count": [
                genuine,
                fraud
            ]
        })

        st.bar_chart(
            chart_data.set_index(
                "Transaction Type"
            )
        )

        # Download
        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except pd.errors.EmptyDataError:
        st.error(
            "❌ The uploaded CSV is empty. "
            "Please upload a CSV containing transaction data."
        )

    except Exception as e:
        st.error("❌ Error processing the file:")
        st.code(str(e))

else:

    st.info(
        "👆 Upload a transaction CSV file to start."
    )

    st.markdown("""
    ### How it works

    **1.** Upload transaction CSV

    **2.** Data preprocessing

    **3.** Machine learning prediction

    **4.** Fraud probability calculation

    **5.** Dashboard statistics

    **6.** Download prediction results
    """)