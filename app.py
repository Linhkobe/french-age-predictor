import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import unicodedata
import numpy as np
import logging
import time
import json
from datetime import date

# --- STEP 1: LOAD ONLY THE WORKING ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("french_age_predictor.h5", compile=False)
    with open("age_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    # Load the scaler 
    import json
    with open("age_scaler.json", "r") as f:
        metadata = json.load(f)
        
    return model, tokenizer, metadata

model, tokenizer, metadata = load_assets()

min_year = metadata["min_"]
max_year = metadata["max_"]

# --- STEP 2: UI DESIGN ---
st.title("French Name Age Predictor")

name_input = st.text_input("First name:", placeholder="e.g., Jean-Kevin")

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

if name_input:
    start_time = time.time()
    # --- STEP 3: PREPROCESS ---
    name_clean = unicodedata.normalize("NFC", name_input.lower().strip())
    seq = tokenizer.texts_to_sequences([name_clean])
    padded = pad_sequences(seq, maxlen=18, padding="post")
    
    # --- STEP 4: PREDICT ---
    # The model outputs a value between 0 and 1
    pred_scaled = model.predict(padded, verbose=0)[0][0]
    
    # --- STEP 5: REVERSE SCALING ---
    # Formula: Year = (Scaled_Value * (Max - Min)) + Min
    final_year = int((pred_scaled * (max_year - min_year)) + min_year)
    
    # Current date
    current_date = date.today()
    
    # Current year
    current_year = current_date.year
    
    # Predicted age = current_year - predicted year
    final_age = current_year - final_year
    duration_time = time.time() - start_time
    
    # logger.info(f"PREDICTION_LOG: name= {name_input}, result= {final_year}, latency = {duration_time: .4f}s")
    log_data = {
        "name": name_input,
        "predicted_year": final_year,
        "predicted_age": final_age,
        "latency": round(duration_time,4),
        "model_version": "v1.0"
    }
    logger.info(json.dumps(log_data, ensure_ascii=False))
    # --- STEP 6: OUTPUT ---
    st.success(f"Prediction: **{final_year}**")
    st.metric(label="Estimated age", value = f"{final_age} years old")