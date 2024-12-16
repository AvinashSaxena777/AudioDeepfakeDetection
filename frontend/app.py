import os
import torch
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from PIL import Image
from models.wav2vec2 import Wav2Vec2Classifier
from models.wavlm import WavLMClassifier
from models.whisper import WhisperClassifier
from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor


# Function to load a model checkpoint and update its weights
# Args:
#   checkpoint_path (str): Path to the checkpoint file.
#   model (torch.nn.Module): Model to load the weights into.
#   model_name (str): Name of the model being loaded.
# Returns:
#   model (torch.nn.Module): Model with loaded weights.
def load_checkpoint(checkpoint_path, model, model_name):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        best_val_accuracy = checkpoint.get('best_val_acc', None)
        if best_val_accuracy is not None:
            st.success(f"Validation Accuracy for model {model_name} : {best_val_accuracy:.4f}")
        return model
    else:
        st.warning("No checkpoint found. Initializing with default weights.")
        return model

# Streamlit page configuration
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

# Main application header
st.markdown('<div class="main-header">Deepfake Audio Detection 🎤</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using WavLM, Wav2Vec2, and Whisper Models</div>', unsafe_allow_html=True)

# Model descriptions section
st.markdown("### Model Descriptions")

# Function to display a resized image with a caption
# Args:
#   image_path (str): Path to the image file.
#   caption (str): Caption for the image.
#   width (int): Width to resize the image.
#   height (int): Height to resize the image.
def display_model_image(image_path, caption, width, height):
    image = Image.open(image_path)
    image = image.resize((width, height))  # Resize the image
    st.image(image, caption=caption, output_format="PNG")

# Define layout with three columns for model descriptions
col1, col2, col3 = st.columns(3)

# Wav2Vec2 Model Description
with col1:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### Wav2Vec2")
    display_model_image("images/wav2vec2.png", "Wav2Vec2", width=600, height=400)
    st.markdown(
        """Wav2Vec2 is a self-supervised model trained on raw audio to learn meaningful representations. It excels in tasks like speech recognition and audio classification.""")
    st.markdown('</div>', unsafe_allow_html=True)

# WavLM Model Description
with col2:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### WavLM")
    display_model_image("images/wavlm.png", "WavLM", width=600, height=400)
    st.markdown(
        """WavLM builds on Wav2Vec2 with additional training on diverse speech datasets. It's effective for noisy environments and multi-speaker scenarios.""")
    st.markdown('</div>', unsafe_allow_html=True)

# Whisper Model Description
with col3:
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### Whisper")
    display_model_image("images/whisper.png", "Whisper", width=600, height=400)
    st.markdown(
        """Whisper is a Transformer-based model by OpenAI for automatic speech recognition and audio classification. It handles multilingual tasks with robust performance.""")
    st.markdown('</div>', unsafe_allow_html=True)

# Upload audio file section
st.markdown("### Upload Your Audio File")
uploaded_file = st.file_uploader("Supported formats: WAV, MP3, FLAC", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Load and visualize the uploaded audio file
    audio, sr = librosa.load(uploaded_file, sr=16000)
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Sample Rate: {sr} Hz &nbsp;&nbsp; | &nbsp;&nbsp; Duration: {len(audio) / sr:.2f} seconds")

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

# Model selection section
st.markdown("### Select a Pretrained Model")
model_options = ["Wav2Vec2", "WavLM", "Whisper"]
cols = st.columns(3)

selected_model = st.session_state.get('selected_model', None)

for i, model in enumerate(model_options):
    button_class = "selected" if selected_model == model else ""
    if cols[i].button(model, key=f"model_{i}", use_container_width=True):
        selected_model = model

# Paths to pretrained model checkpoints
checkpoint_paths = {
    "Wav2Vec2": "./models/wav2vec2/Wav2vec2_best_val_acc_8620.pth",
    "WavLM": "./models/wavlm/WavLM_best_val_acc_9110.pth",
    "Whisper": "./models/whisper/Whisper_best_val_acc_9300.pth"
}

# Function to load a pretrained model with a checkpoint
# Args:
#   model_name (str): Name of the selected model.
# Returns:
#   model (torch.nn.Module): Loaded and initialized model.
@st.cache_resource
def load_model_with_checkpoint(model_name):
    if model_name == "Wav2Vec2":
        model = Wav2Vec2Classifier(model_name="facebook/wav2vec2-base", num_labels=1)
    elif model_name == "WavLM":
        model = WavLMClassifier(model_name="microsoft/wavlm-base", num_labels=1)
    elif model_name == "Whisper":
        model = WhisperClassifier(model_name="openai/whisper-base", num_labels=1)
    checkpoint_path = checkpoint_paths.get(model_name)
    model = load_checkpoint(checkpoint_path, model, model_name)
    model.eval()
    return model

# Function to get feature extractor for the model
# Args:
#   model_name (str): Name of the selected model.
# Returns:
#   feature_extractor (FeatureExtractor): Pretrained feature extractor.
@st.cache_resource
def get_feature_extractor(model_name):
    if model_name == "Wav2Vec2":
        return Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    elif model_name == "WavLM":
        return Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    elif model_name == "Whisper":
        return WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

if uploaded_file and selected_model:
    # Process audio with the selected model
    st.markdown("### Processing Audio with Selected Model...")
    model = load_model_with_checkpoint(selected_model)
    feature_extractor = get_feature_extractor(selected_model)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    if selected_model == "Whisper":
        inputs["input_values"] = inputs.pop("input_features")

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    with torch.no_grad():
        logits = model.forward(inputs["input_values"])
        probs = torch.sigmoid(logits)
        predicted_class = 1 if probs.item() > 0.5 else 0
        confidence = probs.item()

    st.markdown("### Classification Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", "Real" if predicted_class == 0 else "Fake", delta=None)
    with col2:
        if predicted_class == 0:
            confidence = 1 - confidence
        st.metric("Confidence", f"{confidence * 100:.2f} %")

# Footer with attribution
st.markdown(
    """
    <hr style="border:1px solid #eee"/>
    <div style="text-align: center; color: #999;">
    Built with ❤ using <a href="https://streamlit.io/" target="_blank">Streamlit</a> | Powered by <a href="https://huggingface.co/" target="_blank">Hugging Face Transformers</a>
    </div>
    """,
    unsafe_allow_html=True
)
