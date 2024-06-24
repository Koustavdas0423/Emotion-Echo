import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

# Ensure the model path is correct
model_path = 'M:\\Projects\\machineLearning\\Emotion-Echo\\best_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define the scaler
def fit_scaler():
    # Assuming you have access to the training data paths
    crema_path = 'M:\\Projects\\Speech-Emotion-Recognition-main\\datasets\\Crema'
    crema_files = [os.path.join(crema_path, f) for f in os.listdir(crema_path) if f.endswith('.wav')]

    # Extract MFCC features for all files
    X = np.array([extract_mfcc(file) for file in crema_files])

    # Fit the scaler on the training data
    scaler = StandardScaler()
    scaler.fit(X)

    return scaler

scaler = fit_scaler()

def extract_mfcc(filename, n_mfcc=40):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Convert emotions to one-hot encoding
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
enc = OneHotEncoder()
enc.fit(np.array(emotions).reshape(-1, 1))

st.title('Speech Emotion Recognition')
st.write('Upload an audio file to predict the emotion')

uploaded_file = st.file_uploader('Choose an audio file...', type='wav')

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    mfcc = extract_mfcc("temp.wav")
    mfcc = scaler.transform([mfcc])
    mfcc = np.expand_dims(mfcc, axis=1)  # Reshape to (1, 1, features)

    # Predict
    prediction = model.predict(mfcc)
    emotion = np.argmax(prediction)
    emotion_label = enc.categories_[0][emotion]

    st.write(f'The predicted emotion is: {emotion_label}')
