import os
import torch
import onnx
import onnxruntime as ort
from models.wav2vec2 import Wav2Vec2Classifier
from models.wavlm import WavLMClassifier
from models.whisper import WhisperClassifier

# Function to load model checkpoint
def load_checkpoint(checkpoint_path, model):
    """
    Load a model checkpoint and update the model's weights.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, loading default weights.")
    return model

# Function to export model to ONNX format with dynamic axes
def export_to_onnx(model, input_example, export_path, model_name, dynamic_axes):
    """
    Export a PyTorch model to ONNX format with dynamic input axes.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_example (torch.Tensor): Example input tensor to define the model's input shape.
        export_path (str): File path to save the ONNX model.
        model_name (str): Name of the model for logging purposes.
        dynamic_axes (dict): Dictionary defining dynamic axes for input/output tensors.
    """
    model.eval()  # Set the model to evaluation mode
    print(f"Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model,                         # PyTorch model
        input_example,                 # Input example tensor
        export_path,                   # Export path for the ONNX model
        export_params=True,            # Store trained parameters in the model
        opset_version=14,              # ONNX opset version
        do_constant_folding=True,      # Optimize model by folding constants
        input_names=["input"],         # Input tensor name
        output_names=["output"],       # Output tensor name
        dynamic_axes=dynamic_axes      # Allow dynamic axes for input/output
    )
    print(f"Model exported to {export_path}")

# Paths to pretrained model checkpoints
checkpoint_paths = {
    "Wav2Vec2": "./models/wav2vec2/Wav2vec2_best_val_acc_8620.pth",
    "WavLM": "./models/wavlm/WavLM_best_val_acc_9110.pth",
    "Whisper": "./models/whisper/Whisper_best_val_acc_9300.pth"
}

# Dynamic axes configurations
dynamic_axes_wav = {"input": {0: "batch_size", 1: "audio_length"}, "output": {0: "batch_size"}}
dynamic_axes_whisper = {"input": {0: "batch_size", 1: "feature_size", 2: "sequence_length"},
                        "output": {0: "batch_size"}}

# Dummy inputs for ONNX export
dummy_input_wav = torch.randn(1, 16000)  # For Wav2Vec2 and WavLM (1 second of audio sampled at 16kHz)
dummy_input_whisper = torch.randn(1, 80, 3000)  # For Whisper model (batch_size=1, feature_size=80, sequence_length=3000)

# ========================= Wav2Vec2 Model =========================
print("Loading and Exporting Wav2Vec2 Model...")
wav2vec2_model = Wav2Vec2Classifier(model_name="facebook/wav2vec2-base", num_labels=1)
wav2vec2_model = load_checkpoint(checkpoint_paths["Wav2Vec2"], wav2vec2_model)
export_to_onnx(wav2vec2_model, dummy_input_wav, "onnx_models/wav2vec2_model.onnx", "Wav2Vec2", dynamic_axes_wav)

# ========================= WavLM Model =========================
print("Loading and Exporting WavLM Model...")
wavlm_model = WavLMClassifier(model_name="microsoft/wavlm-base", num_labels=1)
wavlm_model = load_checkpoint(checkpoint_paths["WavLM"], wavlm_model)
export_to_onnx(wavlm_model, dummy_input_wav, "onnx_models/wavlm_model.onnx", "WavLM", dynamic_axes_wav)

# ========================= Whisper Model =========================
print("Loading and Exporting Whisper Model...")
whisper_model = WhisperClassifier(model_name="openai/whisper-base", num_labels=1)
whisper_model = load_checkpoint(checkpoint_paths["Whisper"], whisper_model)
export_to_onnx(whisper_model, dummy_input_whisper, "onnx_models/whisper_model.onnx", "Whisper", dynamic_axes_whisper)

print("All models exported successfully to ONNX with dynamic input support!")

# Function to test ONNX Runtime inference
def test_onnx_inference(onnx_path, input_tensor):
    """
    Test the exported ONNX model using ONNX Runtime.

    Args:
        onnx_path (str): Path to the ONNX model file.
        input_tensor (torch.Tensor): Input tensor for testing the model.

    Returns:
        None
    """
    print(f"Testing ONNX model: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    input_data = {"input": input_tensor.numpy()}
    outputs = ort_session.run(None, input_data)
    print("ONNX Runtime Inference Output:", outputs)

# ========================= Test Exported ONNX Models =========================
test_onnx_inference("onnx_models/wav2vec2_model.onnx", torch.randn(2, 20000))  # Batch size=2, audio length=20000
test_onnx_inference("onnx_models/wavlm_model.onnx", torch.randn(3, 18000))  # Batch size=3, audio length=18000
test_onnx_inference("onnx_models/whisper_model.onnx", torch.randn(2, 80, 3000))  # Batch size=2, feature_size=80, sequence_length=2500
