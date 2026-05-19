"""Linear probe models for probing protein language model representations.

Linear probes are simple linear classifiers/regressors trained on frozen PLM embeddings
to assess what information is encoded in different layers of the model.
"""
from typing import Dict
import torch as t
import torch.nn as nn

from project.utils.functions import set_device

# Set the device (GPU/MPS/CPU) for model operations
DEVICE = set_device()

class LinearProbe(nn.Module):
    """
    A linear probe model for classification or regression on protein language model embeddings.
    
    This is a simple single-layer linear model that takes PLM embeddings as input and produces
    task-specific outputs (e.g., class predictions or continuous values). The model is typically
    trained while the PLM remains frozen.
    
    Attributes:
        input_dim: Dimensionality of input embeddings (hidden size of PLM layer)
        output_dim: Number of output classes (for classification) or output dimensions (for regression)
        linear: The linear transformation layer
        loss_fn: Loss function appropriate for the task type
    """
    def __init__(self, input_dim: int, output_dim: int, purpose: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim).to(DEVICE)
        if purpose == 'binary_classification':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif purpose == 'multiclass_classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif purpose == 'regression':
            loss_fn = nn.MSELoss(reduction='none')
        self.loss_fn = loss_fn

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Performs the forward pass through the linear layer.
        """
        return self.linear(x)

    def elementwise_loss(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        """
        Calculates the element-wise loss.
        """
        return self.loss_fn(inputs, targets)

    def average_loss(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        """
        Calculates the average loss.
        """
        # The loss_fn is already configured for `reduction='none'`, so we
        # can take the mean of the element-wise loss.
        return self.elementwise_loss(inputs, targets).mean()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, purpose: str):
        """
        Loads a LinearProbe model from a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
            purpose (str): The purpose of the linear probe (e.g., 'regression').

        Returns:
            LinearProbe: A new instance of LinearProbe with loaded weights.

        Usage:
        probe = LinearProbe.load_from_checkpoint('probe_checkpoint.pth', 'regression')
        """
        # Load the entire checkpoint, which may contain more than just the state_dict
        checkpoint = t.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        # Access the state_dict and extract the relevant linear probe parameters
        state_dict: Dict[str, t.Tensor] = checkpoint.get('state_dict', checkpoint)

        # Determine dimensions from the loaded weights
        try:
            linear_weight = state_dict['linear_probe.linear.weight']
            output_dim, input_dim = linear_weight.shape
        except KeyError:
            raise KeyError("The checkpoint does not contain the expected 'linear_probe.linear.weight' key.")

        # Create the dictionary for the linear layer
        linear_state_dict = {
            'linear.weight': linear_weight,
            'linear.bias': state_dict['linear_probe.linear.bias']
        }

        # Instantiate the model with the inferred dimensions
        linear_model = cls(input_dim, output_dim, purpose)

        # Load the extracted state_dict into the model
        linear_model.load_state_dict(linear_state_dict)
        return linear_model
