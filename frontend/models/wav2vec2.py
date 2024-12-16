import torch
from torch import nn
from transformers import Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2Classifier: A custom PyTorch model for audio classification using Wav2Vec2.
    This model leverages the Wav2Vec2 encoder to extract features from audio input,
    followed by a custom linear layer for binary or multi-class classification.

    Args:
        model_name (str): Name of the pretrained Wav2Vec2 model (e.g., "facebook/wav2vec2-base").
        num_labels (int): Number of output classes (e.g., 2 for binary classification).

    Attributes:
        wav2vec2_model (Wav2Vec2Model): Pretrained Wav2Vec2 model for feature extraction.
        classifier (nn.Linear): A linear layer to map Wav2Vec2 outputs to class logits.
    """

    def __init__(self, model_name: str, num_labels: int):
        super(Wav2Vec2Classifier, self).__init__()

        # Load Wav2Vec2 model (encoder-only, without a classification head)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze Wav2Vec2 model parameters to prevent them from being updated during training
        for param in self.wav2vec2_model.parameters():
            param.requires_grad = False

        # Extract the hidden size from Wav2Vec2 configuration (dimension of encoder output)
        hidden_size = self.wav2vec2_model.config.hidden_size

        # Define a simple classification head: linear layer maps hidden size to num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Wav2Vec2 encoder and classification head.

        Args:
            input_features (torch.Tensor):
                Preprocessed audio features tensor of shape (batch_size, sequence_length).
                This input is typically obtained after preprocessing raw audio with a feature extractor.

        Returns:
            torch.Tensor:
                Logits tensor of shape (batch_size, num_labels), representing raw class scores.
        """
        # Pass the input through Wav2Vec2 to obtain hidden states
        outputs = self.wav2vec2_model(input_features)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Pool the output hidden states along the sequence dimension
        # Here, we compute the mean over the sequence length (average pooling)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Pass the pooled output through the classification head to get logits
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        return logits

    def predict(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Perform inference and return the predicted class index.

        Args:
            input_features (torch.Tensor):
                Preprocessed audio features tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor:
                Predicted class indices of shape (batch_size,).
        """
        # Forward pass to get logits
        logits = self.forward(input_features)  # Shape: (batch_size, num_labels)

        # Apply softmax to logits to get probabilities (optional, for confidence scores)
        probabilities = torch.softmax(logits, dim=-1)  # Shape: (batch_size, num_labels)

        # Get the class with the highest probability (argmax along the last dimension)
        predictions = torch.argmax(probabilities, dim=-1)  # Shape: (batch_size,)

        return predictions
