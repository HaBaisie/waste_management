import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from category_encoders import TargetEncoder
import streamlit as st

# Streamlit App Title
st.title("Waste Collection Prediction App")

# Load dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['Month'] = df['Date'].dt.month
    df['RollingAvgWaste'] = df.groupby('Location of Dumpster')['Volume of Waste Collected (Tonnes)'].transform(
        lambda x: x.rolling(7, min_periods=1).mean())
    df['PrevDayWaste'] = df.groupby('Location of Dumpster')['Volume of Waste Collected (Tonnes)'].shift(1)

    # Label target variable: 1 if waste > 10 tonnes, else 0
    df['WasteOverCapacity'] = (df['Volume of Waste Collected (Tonnes)'] > 6).astype(int)

    # Apply Target Encoding to "Location of Dumpster"
    encoder = TargetEncoder()
    df['Location of Dumpster'] = encoder.fit_transform(df['Location of Dumpster'], df['WasteOverCapacity'])

    # Drop unnecessary columns
    df.drop(columns=['Date', 'Types of Waste', 'Operational Schedule'], inplace=True)

    # Drop rows with NaN values (from shifting)
    df.dropna(inplace=True)

    # Splitting Data
    X = df.drop(columns=['WasteOverCapacity'])  # Features
    y = df['WasteOverCapacity']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Base Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')

    # Train base models
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Generate base model predictions
    rf_proba_train = rf.predict_proba(X_train)
    xgb_proba_train = xgb.predict_proba(X_train)

    # Debug: Check shapes
    st.write("Random Forest predict_proba shape:", rf_proba_train.shape)
    st.write("XGBoost predict_proba shape:", xgb_proba_train.shape)

    # Ensure there are two columns before accessing index 1
    if rf_proba_train.shape[1] == 2:
        rf_preds_train = rf_proba_train[:, 1]
    else:
        rf_preds_train = rf_proba_train[:, 0]  # Fallback to the only column

    if xgb_proba_train.shape[1] == 2:
        xgb_preds_train = xgb_proba_train[:, 1]
    else:
        xgb_preds_train = xgb_proba_train[:, 0]  # Fallback to the only column

    # Repeat the same for test predictions
    rf_proba_test = rf.predict_proba(X_test)
    xgb_proba_test = xgb.predict_proba(X_test)

    if rf_proba_test.shape[1] == 2:
        rf_preds_test = rf_proba_test[:, 1]
    else:
        rf_preds_test = rf_proba_test[:, 0]

    if xgb_proba_test.shape[1] == 2:
        xgb_preds_test = xgb_proba_test[:, 1]
    else:
        xgb_preds_test = xgb_proba_test[:, 0]

    # LSTM Model
    X_train_lstm = np.expand_dims(X_train.values, axis=1)
    X_test_lstm = np.expand_dims(X_test.values, axis=1)

    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(1, X_train.shape[1])),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)

    lstm_preds_train = lstm_model.predict(X_train_lstm).flatten()
    lstm_preds_test = lstm_model.predict(X_test_lstm).flatten()

    # Combine Base Model Predictions for Meta Model
    stacked_train = np.column_stack((rf_preds_train, xgb_preds_train, lstm_preds_train))
    stacked_test = np.column_stack((rf_preds_test, xgb_preds_test, lstm_preds_test))

    # Meta Models
    meta_lr = LogisticRegression()
    meta_xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')

    # Train Meta Models
    meta_lr.fit(stacked_train, y_train)
    meta_xgb.fit(stacked_train, y_train)

    # Predictions from Meta Models
    meta_lr_preds = meta_lr.predict(stacked_test)
    meta_xgb_preds = meta_xgb.predict(stacked_test)

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    st.write("**Logistic Regression Meta Model:**")
    st.write("Precision:", precision_score(y_test, meta_lr_preds))
    st.write("Recall:", recall_score(y_test, meta_lr_preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, meta_lr_preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, meta_lr_preds)))

    st.write("\n**XGBoost Meta Model:**")
    st.write("Precision:", precision_score(y_test, meta_xgb_preds))
    st.write("Recall:", recall_score(y_test, meta_xgb_preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, meta_xgb_preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, meta_xgb_preds)))

    # Predict future waste collection locations
    st.subheader("Predictions on Test Data")
    future_predictions = meta_xgb.predict(stacked_test)
    df_test = pd.DataFrame({'Actual': y_test, 'Predicted': future_predictions})
    st.write(df_test.head())

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=future_predictions, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
