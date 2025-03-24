import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
from category_encoders import TargetEncoder
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Waste Capacity Predictor", page_icon="🗑️", layout="wide")

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background-color: #ffffff;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the saved models and encoder
@st.cache_resource
def load_models():
    try:
        rf_loaded = joblib.load('random_forest_model.pkl')
        xgb_loaded = XGBClassifier()
        xgb_loaded.load_model('xgboost_model.json')
        lstm_model_loaded = tf.keras.models.load_model('lstm_model.h5')
        meta_xgb_loaded = XGBClassifier()
        meta_xgb_loaded.load_model('meta_xgb_model.json')
        encoder_loaded = joblib.load('target_encoder.pkl')
        return rf_loaded, xgb_loaded, lstm_model_loaded, meta_xgb_loaded, encoder_loaded
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Preprocess input data
def preprocess_input(date, encoder, locations):
    try:
        # Create a DataFrame with the input date and all locations
        data = {
            'Date': [date] * len(locations),
            'Location of Dumpster': locations,
            'Frequency of Waste Collection (days)': [7] * len(locations),  # Default value
            'Volume of Waste Collected (Kg)': [5000] * len(locations)  # Default value
        }
        df = pd.DataFrame(data)
        
        # Convert date to datetime and extract features
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        # Apply target encoding to locations
        df['Location of Dumpster'] = encoder.transform(df['Location of Dumpster'])
        
        # Drop the date column
        df.drop(columns=['Date'], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# Make predictions
def predict_waste_over_capacity(input_data, rf_model, xgb_model, lstm_model, meta_model):
    try:
        # Base model predictions
        rf_preds = rf_model.predict_proba(input_data)[:, 1]
        xgb_preds = xgb_model.predict_proba(input_data)[:, 1]
        
        # Reshape input for LSTM
        input_data_lstm = np.expand_dims(input_data.values, axis=1)
        lstm_preds = lstm_model.predict(input_data_lstm).flatten()
        
        # Stack predictions for meta model
        stacked_data = np.column_stack((rf_preds, xgb_preds, lstm_preds))
        
        # Meta model predictions
        predictions = meta_model.predict(stacked_data)
        
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    st.title("🏙️ Waste Capacity Prediction System")
    st.markdown("""
    This app predicts whether dumpsters at various locations will have waste over capacity 
    on a given date based on historical data and machine learning models.
    """)
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # Date input
        input_date = st.date_input(
            "Select a date",
            min_value=datetime.today(),
            value=datetime.today()
        )
        
        # Location input
        locations_input = st.text_area(
            "Enter dumpster locations (one per line or comma-separated)",
            "Location1, Location2, Location3"
        )
        
        # Process locations input
        if "\n" in locations_input:
            locations = [loc.strip() for loc in locations_input.split("\n") if loc.strip()]
        else:
            locations = [loc.strip() for loc in locations_input.split(",") if loc.strip()]
        
        # Example locations button
        if st.button("Load Example Locations"):
            locations = ["Downtown", "Suburb A", "Industrial Zone", "Shopping District", "Residential Area B"]
    
    # Load models
    with st.spinner("Loading prediction models..."):
        rf_model, xgb_model, lstm_model, meta_model, encoder = load_models()
    
    if None in [rf_model, xgb_model, lstm_model, meta_model, encoder]:
        st.error("Failed to load one or more models. Please check the model files.")
        return
    
    # Display input summary
    st.subheader("Input Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Selected Date:** {input_date.strftime('%Y-%m-%d')}")
    with col2:
        st.markdown(f"**Number of Locations:** {len(locations)}")
    
    if len(locations) > 0:
        st.markdown("**Locations to Analyze:**")
        for loc in locations:
            st.markdown(f"- {loc}")
    
    # Make predictions when button is clicked
    if st.button("Predict Waste Capacity", type="primary"):
        if not locations:
            st.warning("Please enter at least one location.")
            return
            
        with st.spinner("Making predictions..."):
            # Preprocess input data
            input_data = preprocess_input(str(input_date), encoder, locations)
            
            if input_data is not None:
                # Make predictions
                predictions = predict_waste_over_capacity(input_data, rf_model, xgb_model, lstm_model, meta_model)
                
                if predictions is not None:
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Create results DataFrame
                    results = pd.DataFrame({
                        'Location': locations,
                        'Waste Over Capacity': ['Yes' if pred == 1 else 'No' for pred in predictions],
                        'Risk Level': ['High' if pred == 1 else 'Low' for pred in predictions]
                    })
                    
                    # Color coding for results
                    def color_risk(val):
                        color = 'red' if val == 'High' else 'green'
                        return f'color: {color}'
                    
                    styled_results = results.style.applymap(color_risk, subset=['Risk Level'])
                    
                    # Display styled table
                    st.dataframe(styled_results, hide_index=True, use_container_width=True)
                    
                    # Summary statistics
                    num_at_risk = sum(predictions)
                    st.metric("Total Locations at Risk", f"{num_at_risk} / {len(locations)}")
                    
                    # Show warning if any locations are at risk
                    if num_at_risk > 0:
                        st.warning(f"⚠️ {num_at_risk} location(s) are predicted to have waste over capacity. Please schedule additional collections.")
                    
                    # Download button for results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"waste_predictions_{input_date}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
