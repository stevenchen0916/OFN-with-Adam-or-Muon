import torch
import deepxde as dde
import deepxde.nn.activations as activations

from .orthogonal_flow_nn import OrthoFlowNN

class QuantumOrthogonalFlowNetwork(dde.nn.pytorch.NN):
    """
    Quantum Orthogonal Flow Network.

    This model follows a multi-input branch--trunk architecture. Each input
    function is encoded by an independent orthogonal-flow branch encoder,
    while query locations are encoded by a trunk encoder. The final operator
    output is obtained by combining the branch embeddings and taking an inner
    product with the trunk embedding.

    The orthogonal-flow encoders are built from quantum-compatible orthogonal
    layers, so the learned transformations can be simulated by RBS-based
    quantum circuits under unary encoding.
    """

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        regularization=None,
        branch_merge="mul",
        trunk_last_activation=False,
        *,
        flow=True,
    ):
        super().__init__()

        if isinstance(activation, dict):
            activation_branch1 = activations.get(activation["branch1"])
            activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch1 = activation_branch2 = self.activation_trunk = activations.get(activation)

        self.branch1 = OrthoFlowNN(layer_sizes_branch1, activation_branch1, flow=flow)
        self.branch2 = OrthoFlowNN(layer_sizes_branch2, activation_branch2, flow=flow)
        self.trunk = OrthoFlowNN(layer_sizes_trunk, self.activation_trunk, flow=flow)

        self.branch_merge = branch_merge
        self.trunk_last_activation = trunk_last_activation

        self.b = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.regularizer = regularization

        self._input_transform = None
        self._output_transform = None

    def merge_branch_embeddings(self, y_branch1, y_branch2):
        if self.branch_merge == "cat":
            return torch.cat((y_branch1, y_branch2), dim=1)

        if y_branch1.shape[-1] != y_branch2.shape[-1]:
            raise ValueError(
                "Branch embedding dimension mismatch: "
                f"branch1={y_branch1.shape[-1]}, branch2={y_branch2.shape[-1]}."
            )

        if self.branch_merge == "add":
            return y_branch1 + y_branch2

        if self.branch_merge == "mul":
            return y_branch1 * y_branch2

        raise ValueError(f"Unsupported branch_merge: {self.branch_merge}")

    def forward(self, inputs):
        x_branch1 = inputs[0]
        x_branch2 = inputs[1]
        x_trunk = inputs[2]

        y_branch1 = self.branch1(x_branch1)
        y_branch2 = self.branch2(x_branch2)

        y_branch = self.merge_branch_embeddings(y_branch1, y_branch2)

        if self._input_transform is not None:
            x_trunk = self._input_transform(x_trunk)

        y_trunk = self.trunk(x_trunk)

        if self.trunk_last_activation:
            y_trunk = self.activation_trunk(y_trunk)

        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise ValueError(
                "Branch and trunk embedding dimension mismatch: "
                f"branch={y_branch.shape[-1]}, trunk={y_trunk.shape[-1]}."
            )

        y = torch.einsum("ip,jp->ij", y_branch, y_trunk)
        y = y + self.b

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)

        return y


QOFN = QuantumOrthogonalFlowNetwork

