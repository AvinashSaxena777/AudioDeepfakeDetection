from transformers import WhisperModel
import torch
import torch.nn as nn


class WhisperClassifier(nn.Module):
    """
    A classification model that uses the encoder of a pre-trained Whisper model
    for audio classification tasks.

    The Whisper model's encoder processes the input audio features to produce
    hidden states, which are pooled and passed through a linear classifier to
    produce logits for classification.

    Attributes:
        whisper_model (WhisperModel): Pre-trained Whisper model with only the encoder.
        classifier (nn.Linear): Linear layer that maps pooled encoder outputs to class logits.

    Args:
        model_name (str): The name or path of the pre-trained Whisper model.
        num_labels (int): The number of target classes for classification.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes the WhisperClassifier model.

        Args:
            model_name (str): Name of the pre-trained Whisper model from Hugging Face.
            num_labels (int): Number of output classes for classification.
        """
        super(WhisperClassifier, self).__init__()

        # Load the Whisper model (encoder only, no generation head)
        self.whisper_model = WhisperModel.from_pretrained(model_name)

        # Freeze the Whisper model parameters to prevent them from being updated during training
        for param in self.whisper_model.parameters():
            param.requires_grad = False

        # Retrieve the hidden size of the Whisper encoder (d_model configuration)
        hidden_size = self.whisper_model.config.d_model

        # Define a simple classification head: a linear layer for output logits
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_features (torch.Tensor): Input audio features of shape (batch_size, seq_len, feature_dim).
                These are typically preprocessed audio inputs suitable for the Whisper encoder.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_labels), representing class scores.
        """
        # Pass inputs through the Whisper encoder to obtain hidden states
        outputs = self.whisper_model.encoder(input_features)

        # Extract the last hidden state from the encoder outputs
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Pool the hidden states across the sequence length (mean pooling)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Pass the pooled output through the classification head to obtain logits
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        return logits
