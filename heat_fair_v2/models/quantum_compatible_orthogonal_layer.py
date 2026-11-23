import numpy as np
import torch
import torch.nn.functional as F

class QuantumCompatibleOrthogonalLayer(torch.nn.Module):
    """
    Quantum-compatible orthogonal layer based on a pyramidal sequence of
    two-dimensional rotations.

    The layer parameterizes a rectangular orthogonal transformation using
    a sequence of pairwise Givens rotations. This structure is compatible
    with RBS-based quantum circuits under unary encoding.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, default=True
        Whether to include an additive bias term.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        num_thetas = self._num_rotation_parameters(
            self.in_features,
            self.out_features,
        )

        self.thetas = torch.nn.Parameter(torch.randn(num_thetas))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        starts, ends, sizes = self._build_rotation_schedule(
            self.in_features,
            self.out_features,
        )

        theta_offsets = [0]
        offset = 0

        for size in sizes:
            offset += size // 2
            theta_offsets.append(offset)

        self.starts = starts
        self.ends = ends
        self.theta_starts = theta_offsets[:-1]
        self.theta_ends = theta_offsets[1:]

    @staticmethod
    def _num_rotation_parameters(in_features, out_features):
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        return int((2 * larger_features - 1 - smaller_features) * smaller_features // 2)

    @staticmethod
    def _build_rotation_schedule(in_features, out_features):
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        if larger_features == smaller_features:
            smaller_features -= 1

        end_indices = np.concatenate(
            [
                np.arange(2, larger_features + 1),
                larger_features + 1 - np.arange(2, smaller_features + 1),
            ]
        )

        start_indices = np.concatenate(
            [
                np.arange(end_indices.shape[0] + smaller_features - larger_features) % 2,
                np.arange(larger_features - smaller_features),
            ]
        )

        slice_sizes = end_indices - start_indices

        if in_features < out_features:
            end_indices = end_indices[::-1].copy()
            start_indices = start_indices[::-1].copy()
            slice_sizes = slice_sizes[::-1].copy()

        return (
            start_indices.tolist(),
            end_indices.tolist(),
            slice_sizes.tolist(),
        )

    @staticmethod
    def _apply_pairwise_rotations(x_slice, cos_values, sin_values):
        batch_size = x_slice.shape[0]
        x_pairs = x_slice.view(batch_size, -1, 2)

        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        rotated = torch.empty_like(x_pairs)
        rotated[..., 0] = x0 * cos_values - x1 * sin_values
        rotated[..., 1] = x0 * sin_values + x1 * cos_values

        return rotated.view(batch_size, -1)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D input tensor of shape (batch_size, in_features), "
                f"but got shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.in_features}, "
                f"got {x.shape[1]}."
            )

        # Compose the same stage-wise rotations on a small identity matrix.
        # Applying them to every sample separately creates large autograd graphs.
        # The parameterization, stage order and padding convention are unchanged.
        inputs = x
        width = max(self.in_features, self.out_features)
        x = torch.eye(width, dtype=x.dtype, device=x.device)

        cos_all = torch.cos(self.thetas)
        sin_all = torch.sin(self.thetas)

        for start, end, theta_start, theta_end in zip(
            self.starts,
            self.ends,
            self.theta_starts,
            self.theta_ends,
        ):
            x_slice = x[:, start:end]

            cos_values = cos_all[theta_start:theta_end].view(1, -1)
            sin_values = sin_all[theta_start:theta_end].view(1, -1)

            rotated = self._apply_pairwise_rotations(
                x_slice,
                cos_values,
                sin_values,
            )

            x = torch.slice_scatter(
                x,
                rotated,
                dim=1,
                start=start,
                end=end,
            )

        if self.in_features < self.out_features:
            x = x[-self.in_features:, :]
        if self.in_features > self.out_features:
            x = x[:, self.in_features - self.out_features:]

        x = inputs @ x

        if self.bias is not None:
            x = x + self.bias

        return x
