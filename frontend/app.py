import os
import torch
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor
import onnxruntime as ort

# -----------------------------------------
# Streamlit Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎤",
    layout="wide"
)

# Inject custom CSS styles for better UI appearance
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4F8EF7;
        text-align: center;
    }
    .description {
        font-size: 1.1rem;
        color: #666;
        margin: 10px 0;
    }
    .card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    hr {
        border: 0;
        border-top: 1px solid #eee;
    }
    div.stButton > button {
        background-color: #f0f0f0;
        color: #333333;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #e0e0e0;
    }
    div.stButton > button:active {
        background-color: #4F8EF7;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------
# Main Application Header
# -----------------------------------------
st.markdown('<div class="main-header">Deepfake Audio Detection 🎤</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using WavLM, Wav2Vec2, and Whisper Models</div>', unsafe_allow_html=True)

# -----------------------------------------
# Model Descriptions Section
# -----------------------------------------
st.markdown("### Model Descriptions")


def display_model_image(image_path: str, caption: str, width: int, height: int):
    """
    Display a resized image with a caption in Streamlit.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    caption : str
        Caption for the image.
    width : int
        Width to resize the image to.
    height : int
        Height to resize the image to.
    """
    image = Image.open(image_path)
    image = image.resize((width, height))  # Resize the image
    st.image(image, caption=caption, output_format="PNG")


# Define layout with three columns for model descriptions
col1, col2, col3 = st.columns(3)

# -----------------------------------------
# Wav2Vec2 Model Description
# -----------------------------------------
with col1:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### Wav2Vec2")
    display_model_image("images/wav2vec2.png", "Wav2Vec2", width=600, height=400)
    st.markdown(
        """Wav2Vec2 is a self-supervised model trained on raw audio to learn meaningful representations. It excels in tasks like speech recognition and audio classification.""")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# WavLM Model Description
# -----------------------------------------
with col2:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### WavLM")
    display_model_image("images/wavlm.png", "WavLM", width=600, height=400)
    st.markdown(
        """WavLM builds on Wav2Vec2 with additional training on diverse speech datasets. It's effective for noisy environments and multi-speaker scenarios.""")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# Whisper Model Description
# -----------------------------------------
with col3:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### Whisper")
    display_model_image("images/whisper.png", "Whisper", width=600, height=400)
    st.markdown(
        """Whisper is a Transformer-based model by OpenAI for automatic speech recognition and audio classification. It handles multilingual tasks with robust performance.""")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# Audio File Upload Section
# -----------------------------------------
st.markdown("### Upload Your Audio File")
uploaded_file = st.file_uploader("Supported formats: WAV, MP3, FLAC", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Load the uploaded audio file at 16kHz sampling rate
    audio, sr = librosa.load(uploaded_file, sr=16000)

    # Display audio player in the browser
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Sample Rate: {sr} Hz &nbsp;&nbsp;|&nbsp;&nbsp; Duration: {len(audio) / sr:.2f} seconds")

    # Expandable section to visualize the audio waveform and mel spectrogram
    with st.expander("Audio Visualization", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Raw Waveform")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(np.linspace(0, len(audio) / sr, len(audio)), audio, color="#4F8EF7")
            ax.set_title("Raw Audio Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Mel Spectrogram")
            fig, ax = plt.subplots(figsize=(8, 3))
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            img = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title("Mel Spectrogram")
            st.pyplot(fig)

# -----------------------------------------
# Model Selection Section
# -----------------------------------------
st.markdown("### Select a Pretrained Model")
model_options = ["Wav2Vec2", "WavLM", "Whisper"]
cols = st.columns(3)

selected_model = st.session_state.get('selected_model', None)

# Display model selection buttons
for i, model in enumerate(model_options):
    if cols[i].button(model, key=f"model_{i}", use_container_width=True):
        selected_model = model


@st.cache_resource
def load_model_with_onnx_runtime(model_name: str):
    """
    Load an ONNX model into an ONNX Runtime inference session.

    Parameters
    ----------
    model_name : str
        The name of the selected model ("Wav2Vec2", "WavLM", or "Whisper").

    Returns
    -------
    ort.InferenceSession
        The ONNX Runtime inference session for the selected model.
    """
    if model_name == "Wav2Vec2":
        ort_session = ort.InferenceSession("onnx_models/wav2vec2_model.onnx")
    elif model_name == "WavLM":
        ort_session = ort.InferenceSession("onnx_models/wavlm_model.onnx")
    elif model_name == "Whisper":
        ort_session = ort.InferenceSession("onnx_models/whisper_model.onnx")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return ort_session


@st.cache_resource
def get_feature_extractor(model_name: str):
    """
    Retrieve the appropriate Hugging Face feature extractor for the given model.

    Parameters
    ----------
    model_name : str
        The name of the selected model ("Wav2Vec2", "WavLM", or "Whisper").

    Returns
    -------
    transformers.FeatureExtractor
        A feature extractor configured for the specified model.
    """
    if model_name == "Wav2Vec2":
        return Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    elif model_name == "WavLM":
        return Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    elif model_name == "Whisper":
        return WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


# -----------------------------------------
# Inference and Results Section
# -----------------------------------------
if uploaded_file and selected_model:
    st.markdown("### Processing Audio with Selected Model...")

    # Extract features from the uploaded audio using the selected model's extractor
    feature_extractor = get_feature_extractor(selected_model)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    # For Whisper, rename "input_features" to "input_values" to match ONNX input expectations
    if selected_model == "Whisper":
        inputs["input_values"] = inputs.pop("input_features")

    # Load the model inference session using ONNX Runtime
    ort_session = load_model_with_onnx_runtime(selected_model)

    # Display a progress bar to simulate processing
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Run inference with ONNX model
    input_data = {"input": inputs["input_values"].numpy()}
    logits = ort_session.run(None, input_data)

    # Compute sigmoid probabilities for binary classification (real vs. fake)
    probs = torch.sigmoid(torch.tensor(logits[0]))

    # Threshold at 0.5 for binary classification: >0.5 = fake, <=0.5 = real
    predicted_class = 1 if probs.item() > 0.5 else 0
    confidence = probs.item()

    # -----------------------------------------
    # Display Results
    # -----------------------------------------
    st.markdown("### Classification Results")
    col1, col2 = st.columns(2)
    with col1:
        # Display predicted class
        st.metric("Predicted Class", "Real" if predicted_class == 0 else "Fake")
    with col2:
        # Adjust confidence if the class is "Real", because we defined fake > 0.5
        if predicted_class == 0:
            confidence = 1 - confidence
        st.metric("Confidence", f"{confidence * 100:.2f} %")

# -----------------------------------------
# Footer with Attribution
# -----------------------------------------
st.markdown(
    """
    <hr style="border:1px solid #eee"/>
    <div style="text-align: center; color: #999;">
    Built with ❤ using <a href="https://streamlit.io/" target="_blank">Streamlit</a> | Powered by <a href="https://huggingface.co/" target="_blank">Hugging Face Transformers</a>
    </div>
    """,
    unsafe_allow_html=True
)
