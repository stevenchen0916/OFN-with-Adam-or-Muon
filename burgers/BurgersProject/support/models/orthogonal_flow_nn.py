import torch
import deepxde as dde
import deepxde.nn.activations as activations
from .quantum_compatible_orthogonal_layer import QuantumCompatibleOrthogonalLayer

class OrthoFlowNN(dde.nn.pytorch.NN):
    """
    Orthogonal Flow Neural Network.

    Each hidden layer applies a normalized quantum-compatible orthogonal
    transformation followed by a controlled latent update:
        u_{k+1} = (1 - h_k) x_k + h_k Q_k(x_k),
        x_{k+1} = sigma(u_{k+1}).

    Here Q_k is implemented by QuantumCompatibleOrthogonalLayer.
    """

    def __init__(self, layer_sizes, activation, *, flow=True):
        super().__init__()

        self.flow = flow
        self.activation = activations.get(activation)
        self.layer_sizes = list(layer_sizes)

        if len(self.layer_sizes) < 2:
            raise ValueError(f"Invalid layer_sizes: {layer_sizes}")

        self.hidden_layers = torch.nn.ModuleList()

        for i in range(len(self.layer_sizes) - 2):
            self.hidden_layers.append(
                QuantumCompatibleOrthogonalLayer(
                    self.layer_sizes[i],
                    self.layer_sizes[i + 1],
                )
            )

        self.h_params = torch.nn.ParameterList()
        if self.flow:
            self.h_params.extend(
                torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                for _ in self.hidden_layers
            )

        self.output_layer = torch.nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
        )

        self._input_transform = None
        self._output_transform = None

    def hidden_preactivation(self, x, index):
        x = x.float()
        norm = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z = self.hidden_layers[index](x / norm)
        if self.flow and x.shape[1] == z.shape[1]:
            h = torch.sigmoid(self.h_params[index])
            return (1.0 - h) * x + h * z
        return z

    def forward(self, inputs):
        x = inputs

        if self._input_transform is not None:
            x = self._input_transform(x)

        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_preactivation(x, i))

        x = self.output_layer(x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x
