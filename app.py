import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Telco Churn App", layout="wide")

st.title("📊 Telco Customer Churn Prediction System")


expected_cols = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
    'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
    'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
    'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
    'monthlycharges', 'totalcharges'
]

categorical_cols = [
    'gender','partner','dependents','phoneservice','multiplelines',
    'internetservice','onlinesecurity','onlinebackup','deviceprotection',
    'techsupport','streamingtv','streamingmovies','contract',
    'paperlessbilling','paymentmethod'
]


def fit_encode(df):
    local_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        local_encoders[col] = le
    return df, local_encoders

def safe_decode_and_encode(df, encoders_dict):
    df_display = df.copy()
    df_model = df.copy()
    
    
    if 'seniorcitizen' in df_display.columns:
        df_display['seniorcitizen'] = df_display['seniorcitizen'].astype(str).apply(
            lambda x: "Yes" if x.strip() in ['1', '1.0', 'Yes'] else "No"
        )
    
    for col in categorical_cols:
        known_classes = list(encoders_dict[col].classes_)
        
        def to_words(val):
            val_str = str(val).strip().split('.')[0]
            if val_str.isdigit():
                idx = int(val_str)
                if 0 <= idx < len(known_classes):
                    return known_classes[idx]
            return val

        df_display[col] = df_display[col].apply(to_words)
        
        df_model[col] = df_display[col].astype(str)
        known = encoders_dict[col].classes_
        df_model[col] = df_model[col].apply(lambda x: x if x in known else known[0])
        df_model[col] = encoders_dict[col].transform(df_model[col])
        
    
    if 'seniorcitizen' in df_model.columns:
        df_model['seniorcitizen'] = df_model['seniorcitizen'].astype(str).apply(
            lambda x: 1 if x.strip() in ['1', '1.0', 'Yes'] else 0
        )
        
    return df_display, df_model


@st.cache_resource
def train_and_cache_system():
    df = pd.read_csv("Telco_churn.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates()

    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df = df.dropna()

    if 'customerid' in df.columns:
        df = df.drop(columns=['customerid'])

    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    df, trained_encoders = fit_encode(df)

    X = df[expected_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return {"model": model, "accuracy": acc, "encoders": trained_encoders}

system_artifacts = train_and_cache_system()
model = system_artifacts["model"]
accuracy = system_artifacts["accuracy"]
encoders = system_artifacts["encoders"]


mode = st.sidebar.selectbox("Select Mode", ["Single Prediction", "Bulk Prediction"])


if mode == "Bulk Prediction":
    st.header("📂 Bulk Prediction")
    files = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=True)

    if files:
        for file in files:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip().str.lower()
            df = df.drop_duplicates()

            df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
            df = df.dropna()

            if 'customerid' in df.columns:
                df = df.drop(columns=['customerid'])
            if 'churn' in df.columns:
                df = df.drop(columns=['churn'])

            for col in expected_cols:
                if col not in df.columns:
                    if col == 'seniorcitizen':
                        df[col] = 0
                    else:
                        df[col] = "No"

            df = df[expected_cols]

            
            df_clean, df_encoded = safe_decode_and_encode(df, encoders)

            preds = model.predict(df_encoded)
            
            df_clean["Prediction"] = ["Churn" if p == 1 else "No Churn" for p in preds]

            st.subheader(file.name)
            st.dataframe(df_clean)  # Shows "Yes" / "No" for senior citizens along with other clean text words!

            st.download_button(
                "Download Result",
                df_clean.to_csv(index=False).encode(),
                file.name + "_result.csv"
            )


else:
    st.header("👤 Single Prediction")
    st.info(f"Model Accuracy: {accuracy*100:.2f}%")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male","Female"])
        seniorcitizen = st.selectbox("Senior Citizen", ["No", "Yes"]) # Changed layout drop-down to Yes/No
        partner = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.number_input("Tenure", 0, 100, 10)
        phoneservice = st.selectbox("Phone Service", ["Yes","No"])

    with col2:
        multiplelines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        internetservice = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
        onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    with col3:
        streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperlessbilling = st.selectbox("Paperless Billing", ["Yes","No"])
        paymentmethod = st.selectbox(
            "Payment Method",
            ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
        )
        monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    totalcharges = float(tenure * monthlycharges) if tenure > 0 else 0.0

    input_df = pd.DataFrame([{
        'gender': gender, 'seniorcitizen': seniorcitizen, 'partner': partner,
        'dependents': dependents, 'tenure': int(tenure), 'phoneservice': phoneservice,
        'multiplelines': multiplelines, 'internetservice': internetservice,
        'onlinesecurity': onlinesecurity, 'onlinebackup': onlinebackup,
        'deviceprotection': deviceprotection, 'techsupport': techsupport,
        'streamingtv': streamingtv, 'streamingmovies': streamingmovies,
        'contract': contract, 'paperlessbilling': paperlessbilling,
        'paymentmethod': paymentmethod, 'monthlycharges': float(monthlycharges),
        'totalcharges': float(totalcharges)
    }])

    if st.button("Predict"):
        _, input_df_encoded = safe_decode_and_encode(input_df, encoders)
        input_df_encoded = input_df_encoded.reindex(columns=expected_cols, fill_value=0)

        pred = model.predict(input_df_encoded)[0]
        prob = model.predict_proba(input_df_encoded)[0][1]

        st.markdown("---")

        if pred == 1:
            st.error("🚨 CHURN")
        else:
            st.success("✅ NOT CHURN")

        st.metric("Churn Probability", f"{prob*100:.2f}%")
