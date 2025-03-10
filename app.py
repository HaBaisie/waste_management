import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("new.csv")
    return df

# Preprocess data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['RollingAvgWaste'] = df.groupby('Location of Dumpster')['Volume of Waste Collected (Tonnes)'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['PrevDayWaste'] = df.groupby('Location of Dumpster')['Volume of Waste Collected (Tonnes)'].shift(1)
    df['WasteOverCapacity'] = (df['Volume of Waste Collected (Tonnes)'] > 5).astype(int)
    encoder = TargetEncoder()
    df['Location of Dumpster'] = encoder.fit_transform(df['Location of Dumpster'], df['WasteOverCapacity'])
    df.drop(columns=['Date', 'Types of Waste', 'Operational Schedule'], inplace=True)
    df.dropna(inplace=True)
    return df, encoder

# Train models
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return rf, xgb

# LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function
def main():
    st.title("Waste Collection Prediction App")
    
    # Load data
    df = load_data()
    
    # Preprocess data
    df, encoder = preprocess_data(df)
    
    # Splitting Data
    X = df.drop(columns=['WasteOverCapacity'])
    y = df['WasteOverCapacity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    # Train base models
    rf, xgb = train_models(X_train, y_train)
    
    # LSTM Model
    X_train_lstm = np.expand_dims(X_train.values, axis=1)
    X_test_lstm = np.expand_dims(X_test.values, axis=1)
    lstm_model = create_lstm_model((1, X_train.shape[1]))
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Combine Base Model Predictions for Meta Model
    rf_preds_train = rf.predict_proba(X_train)[:, 1]
    xgb_preds_train = xgb.predict_proba(X_train)[:, 1]
    lstm_preds_train = lstm_model.predict(X_train_lstm).flatten()
    stacked_train = np.column_stack((rf_preds_train, xgb_preds_train, lstm_preds_train))
    
    # Train Meta Models
    meta_xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    meta_xgb.fit(stacked_train, y_train)
    
    # User input for prediction
    st.subheader("Predict Full Locations")
    input_date = st.date_input("Enter a date to predict full locations:")
    
    if st.button("Predict"):
        # Create a DataFrame for the input date
        input_df = pd.DataFrame({
            'Date': [input_date],
            'Location of Dumpster': df['Location of Dumpster'].unique()[0]  # Placeholder, will be encoded
        })
        
        # Preprocess the input data
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df['DayOfWeek'] = input_df['Date'].dt.dayofweek
        input_df['Month'] = input_df['Date'].dt.month
        input_df['RollingAvgWaste'] = df['RollingAvgWaste'].mean()  # Use mean as placeholder
        input_df['PrevDayWaste'] = df['PrevDayWaste'].mean()  # Use mean as placeholder
        
        # Encode 'Location of Dumpster'
        input_df['Location of Dumpster'] = encoder.transform(input_df['Location of Dumpster'])
        
        # Drop unnecessary columns
        input_df.drop(columns=['Date'], inplace=True)
        
        # Generate predictions for all locations
        predictions = []
        for location in df['Location of Dumpster'].unique():
            input_df['Location of Dumpster'] = encoder.transform([location])[0]
            X_input = input_df.values
            
            # Base model predictions
            rf_pred = rf.predict_proba(X_input)[:, 1][0]
            xgb_pred = xgb.predict_proba(X_input)[:, 1][0]
            lstm_pred = lstm_model.predict(np.expand_dims(X_input, axis=1)).flatten()[0]
            
            # Meta model prediction
            stacked_input = np.column_stack((rf_pred, xgb_pred, lstm_pred))
            meta_pred = meta_xgb.predict(stacked_input)[0]
            
            predictions.append((location, meta_pred))
        
        # Display predictions
        st.subheader("Predicted Full Locations")
        for location, pred in predictions:
            if pred == 1:
                st.write(f"Location {location} is likely to be full on {input_date}.")

if __name__ == "__main__":
    main()