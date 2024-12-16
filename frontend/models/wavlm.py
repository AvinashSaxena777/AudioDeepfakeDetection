from transformers import WavLMModel
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model definition for WavLM based deepfake audio detection
class WavLMClassifier(nn.Module):
    """
    A custom audio classifier using WavLM's encoder for binary classification.
    This model utilizes WavLM as a feature extractor and adds a simple linear layer as a classification head.

    Args:
        model_name (str): Name of the WavLM model from Hugging Face model hub.
        num_labels (int): Number of output labels for classification.

    Attributes:
        wavlm_model (WavLMModel): Pretrained WavLM model for audio feature extraction.
        classifier (nn.Linear): A linear layer mapping hidden representations to class logits.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes the WavLMClassifier.

        Args:
            model_name (str): Name of the WavLM model from Hugging Face model hub.
            num_labels (int): Number of output labels for classification.
        """
        super(WavLMClassifier, self).__init__()

        # Load WavLM model without the classification head
        self.wavlm_model = WavLMModel.from_pretrained(model_name)

        # Optional: Freeze WavLM model parameters to prevent updates during training
        for param in self.wavlm_model.parameters():
            param.requires_grad = False

        # Get the hidden size of the WavLM encoder to define the classifier input size
        hidden_size = self.wavlm_model.config.hidden_size

        # Classification head: a linear layer mapping from hidden_size to num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the WavLMClassifier.

        Args:
            input_values (torch.Tensor):
                Preprocessed audio features tensor of shape (batch_size, seq_len).
                This input is typically raw audio data converted to tensors.

        Returns:
            torch.Tensor:
                Logits tensor of shape (batch_size, num_labels), representing class scores.
        """
        # Pass inputs through WavLM encoder to get hidden states
        outputs = self.wavlm_model(input_values)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Pool the hidden states across the sequence dimension (e.g., take the mean)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Pass pooled output through the classifier to get logits
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)

        return logits

    def predict(self, input_values: torch.Tensor) -> int:
        """
        Makes a binary prediction based on the preprocessed audio input.

        Args:
            input_values (torch.Tensor):
                Preprocessed audio features tensor of shape (batch_size, seq_len).

        Returns:
            int:
                Predicted class label (0 or 1).
        """
        # Perform a forward pass to get logits
        logits = self.forward(input_values)  # Shape: (batch_size, num_labels)

        # Apply sigmoid to logits to convert them into probabilities
        probabilities = torch.sigmoid(logits)  # Shape: (batch_size, num_labels)

        # Threshold the probabilities to determine the class (default threshold = 0.5)
        predicted_class = (probabilities > 0.5).long().item()

        return predicted_class
