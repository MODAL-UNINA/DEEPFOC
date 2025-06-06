#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from .utils import conjugate_torch
from .PT_axis import FocalMechanism_torch

# %%


class KaganAngle(pl.LightningModule):
    """
    Computes the Kagan angle, which measures the rotation between two moment tensors
    defined by their strike, dip, and rake angles.

    Reference:
        Kagan, Y. "Simplified algorithms for calculating double-couple rotation",
        Geophysical Journal, Volume 171, Issue 1, pp. 411-418.
    """

    def __init__(
        self,
        reduction: Union[str, None] = None,
        normalize: bool = False,
        momentum_loss: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the KaganAngle module.

        Args:
            reduction (str, optional): Reduction mode for output tensor values ('mean', 'max',
                                    'min', 'sum', or 'std'). If None, no reduction is applied.
            normalize (bool, optional): If True, normalizes the computed Kagan angle by 120°.
            momentum_loss (bool, optional): If True, computes the momentum similarity loss.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(KaganAngle, self).__init__(*args, **kwargs)
        self.reduction = reduction
        self.momentum_loss = momentum_loss
        self.normalize = normalize

    def forward(
        self,
        strike_dip_rake_pred: Union[torch.Tensor, list, np.ndarray],
        batch: Union[torch.Tensor, dict, list, np.ndarray],
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Kagan angle between two moment tensors defined by their strike, dip, and rake.

        Args:
            strike_dip_rake_pred (torch.Tensor or list or np.ndarray):
                Predicted moment tensor parameters as [strike, dip, rake].
            batch (torch.Tensor or dict or list or np.ndarray):
                Ground-truth moment tensor parameters. If a dict is provided, it must contain the key
                "SDR" corresponding to the parameters.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Optionally, a 'device' key may be provided to move
                    tensors to a specified device.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - "Kagan" (torch.Tensor): The computed Kagan angle (in degrees, possibly normalized).
                - "Momentum" (torch.Tensor, optional): The momentum similarity loss (if momentum_loss is enabled).
        """
        # Convert predicted parameters to a torch.Tensor if they are provided as a list or numpy array.
        if isinstance(strike_dip_rake_pred, list | np.ndarray):
            strike_dip_rake_pred = torch.tensor(strike_dip_rake_pred)

        # Extract ground-truth parameters: if a dict is provided, use the "SDR" key; otherwise, convert to a tensor.
        if isinstance(batch, list | np.ndarray):
            strike_dip_rake_true = torch.tensor(batch)
        elif isinstance(batch, dict):
            strike_dip_rake_true = batch["SDR"]
        elif isinstance(batch, torch.Tensor):
            strike_dip_rake_true = batch
        else:
            raise ValueError(
                "Batch should be a list, dict with 'SDR' key or torch.Tensor"
            )

        # Ensure that the last dimension contains 3 values (strike, dip, rake).
        assert (
            strike_dip_rake_pred.shape[-1] == 3
        ), "Strike, dip, rake should have 3 values"
        assert (
            strike_dip_rake_true.shape[-1] == 3
        ), "Strike, dip, rake should have 3 values"

        # If a device is specified in kwargs, move the tensors to that device.
        if "device" in kwargs:
            strike_dip_rake_pred = strike_dip_rake_pred.to(kwargs["device"])
            strike_dip_rake_true = strike_dip_rake_true.to(kwargs["device"])

        # Convert to float64 for improved numerical precision.
        strike_dip_rake_pred = strike_dip_rake_pred.to(torch.float64)
        strike_dip_rake_true = strike_dip_rake_true.to(torch.float64)

        # Convert the strike, dip, and rake parameters to moment tensor matrices.
        tensor1 = self.plane_to_torchtensor(*torch.unbind(strike_dip_rake_pred, dim=-1))
        tensor2 = self.plane_to_torchtensor(*torch.unbind(strike_dip_rake_true, dim=-1))

        # Compute the Kagan angle between the two moment tensor matrices.
        kagan = self.calc_theta(tensor1, tensor2)
        kagan = kagan.to(torch.float32)  # Cast the result back to float32.

        # Optionally normalize the Kagan angle by 120°.
        if self.normalize:
            kagan = kagan / 120.0

        output: Dict[str, torch.Tensor] = {"Kagan": kagan}

        # Compute the momentum similarity loss if enabled.
        if self.momentum_loss:
            # Flatten the moment tensor matrices, preserving the batch dimension.
            momentum_cosine = F.cosine_similarity(
                tensor1.flatten(start_dim=1), tensor2.flatten(start_dim=1), dim=-1
            )
            # Momentum loss is defined as one minus the cosine similarity.
            momentum_loss = 1 - momentum_cosine
            output["Momentum"] = momentum_loss

        # Apply reduction to the output values if a reduction mode is specified.
        if self.reduction == "mean":
            output = {k: v.mean() for k, v in output.items()}
        elif self.reduction == "max":
            output = {k: v.max() for k, v in output.items()}
        elif self.reduction == "min":
            output = {k: v.min() for k, v in output.items()}
        elif self.reduction == "sum":
            output = {k: v.sum() for k, v in output.items()}
        elif self.reduction == "std":
            # If multiple outputs exist, compute the standard deviation for each.
            if len(output) > 1:
                output = {k: v.std() for k, v in output.items()}
            else:
                output = torch.zeros(len(output))
        else:
            output = output
        # If no reduction is specified, the output dictionary remains unchanged.
        return output

    def calc_theta(self, vm1: torch.Tensor, vm2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Kagan angle between two moment tensor matrices.

        The function computes the eigenvector representations for both moment tensors (using
        self.calc_eigenvec) and then determines the minimal rotation angle between these
        representations by considering sign ambiguities.

        Args:
            vm1 (torch.Tensor): Moment tensor matrix derived from predicted parameters.
            vm2 (torch.Tensor): Moment tensor matrix derived from ground-truth parameters.

        Returns:
            torch.Tensor: The Kagan angle in degrees between the two moment tensors.
        """
        # Compute the eigenvectors for both moment tensors.
        V1 = self.calc_eigenvec(vm1)
        V2 = self.calc_eigenvec(vm2)
        # Detach V2 to prevent gradients from flowing through it.
        V2.detach()

        # Calculate the initial angle between the eigenvector representations.
        th = self.ang_from_R1R2(V1, V2)

        # Define negation masks to account for the inherent sign ambiguity in eigenvectors.
        negation_mask = torch.tensor(
            [[-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=V2.dtype, device=V2.device
        )

        # Evaluate the angle for each negated combination and keep the minimum angle.
        for neg_mask in negation_mask:
            V3 = V2.clone() * neg_mask
            x = self.ang_from_R1R2(V1, V3)
            th = torch.min(th, x)

        # Convert the angle from radians to degrees.
        return th * 180.0 / torch.pi

    @staticmethod
    def plane_to_torchtensor(
        strike: Union[float, torch.Tensor],
        dip: Union[float, torch.Tensor],
        rake: Union[float, torch.Tensor],
        mag: Union[float, torch.Tensor] = 2.0,
    ) -> torch.Tensor:
        """
        Convert strike, dip, and rake angles (in degrees) to a moment tensor representation.

        This function takes the strike, dip, and rake angles corresponding to the first nodal plane
        of an earthquake source and computes the 3x3 moment tensor. The seismic moment is derived from
        the magnitude using a scaling relation, and the tensor components are computed using standard
        trigonometric formulas. Finally, the moment tensor is scaled from dyne-cm to N-m.

        Args:
            strike (float or torch.Tensor): Strike angle (in degrees) from the first nodal plane.
            dip (float or torch.Tensor): Dip angle (in degrees) from the first nodal plane.
            rake (float or torch.Tensor): Rake angle (in degrees) from the first nodal plane.
            mag (float or torch.Tensor, optional): Earthquake magnitude used to compute the seismic moment.
                                                Default is 2.0.

        Returns:
            torch.Tensor: A 3x3 tensor representing the moment tensor with components arranged as:
                        [[mrr, mrt, mrp],
                        [mrt, mtt, mtp],
                        [mrp, mtp, mpp]]
        """

        # If the inputs are provided as floats, convert them to torch tensors with gradient tracking enabled.
        if isinstance(strike, float):
            strike = torch.tensor(strike, requires_grad=True)
        if isinstance(dip, float):
            dip = torch.tensor(dip, requires_grad=True)
        if isinstance(rake, float):
            rake = torch.tensor(rake, requires_grad=True)
        if isinstance(mag, float):
            mag = torch.tensor(mag, requires_grad=True)

        # Define conversion factor from degrees to radians.
        d2r = torch.tensor(torch.pi / 180.0)

        # Compute the seismic moment:
        # The scaling relation is given by: magpow = mag * 1.5 + 16.1, so the moment is 10^(magpow).
        magpow = mag * 1.5 + 16.1
        mom = torch.pow(10, magpow)

        # Compute trigonometric functions for the given angles (converted to radians).
        sin_2_dip_d2r = torch.sin(2 * dip * d2r)  # sin(2*dip)
        sin_rake_d2r = torch.sin(rake * d2r)  # sin(rake)
        sin_2_strike_d2r = torch.sin(2 * strike * d2r)  # sin(2*strike)
        sin_strike_d2r = torch.sin(strike * d2r)  # sin(strike)
        cos_strike_d2r = torch.cos(strike * d2r)  # cos(strike)
        cos_2_dip_d2r = torch.cos(2 * dip * d2r)  # cos(2*dip)
        sin_dip_d2r_cos_rake_d2r = torch.sin(dip * d2r) * torch.cos(
            rake * d2r
        )  # sin(dip)*cos(rake)
        cos_dip_d2r_cos_rake_d2r = torch.cos(dip * d2r) * torch.cos(
            rake * d2r
        )  # cos(dip)*cos(rake)

        # Calculate moment tensor components based on the standard formulas.
        mrr = mom * sin_2_dip_d2r * sin_rake_d2r
        mtt = -mom * (
            (sin_dip_d2r_cos_rake_d2r * sin_2_strike_d2r)
            + (sin_2_dip_d2r * sin_rake_d2r * (sin_strike_d2r * sin_strike_d2r))
        )
        mpp = mom * (
            (sin_dip_d2r_cos_rake_d2r * sin_2_strike_d2r)
            - (sin_2_dip_d2r * sin_rake_d2r * (cos_strike_d2r * cos_strike_d2r))
        )
        mrt = -mom * (
            (cos_dip_d2r_cos_rake_d2r * cos_strike_d2r)
            + (cos_2_dip_d2r * sin_rake_d2r * sin_strike_d2r)
        )
        mrp = mom * (
            (cos_dip_d2r_cos_rake_d2r * sin_strike_d2r)
            - (cos_2_dip_d2r * sin_rake_d2r * cos_strike_d2r)
        )
        mtp = -mom * (
            (sin_dip_d2r_cos_rake_d2r * torch.cos(2 * strike * d2r))
            + (0.5 * sin_2_dip_d2r * sin_rake_d2r * sin_2_strike_d2r)
        )

        # Assemble the moment tensor matrix by stacking the computed components.
        # The resulting matrix has the structure:
        # [[mrr, mrt, mrp],
        #  [mrt, mtt, mtp],
        #  [mrp, mtp, mpp]]
        mt_matrix = torch.stack(
            [
                torch.stack([mrr, mrt, mrp], dim=-1),
                torch.stack([mrt, mtt, mtp], dim=-1),
                torch.stack([mrp, mtp, mpp], dim=-1),
            ],
            dim=-1,
        )

        # Convert the units from dyne-cm to N-m (scaling factor of 1e-7).
        mt_matrix = mt_matrix * 1e-7
        return mt_matrix

    @staticmethod
    def calc_eigenvec(TM: torch.Tensor) -> torch.Tensor:
        """
        Calculate the eigenvector representation of a moment tensor matrix.

        This function performs an eigen decomposition on the given symmetric 3x3 moment tensor using
        torch.linalg.eigh, which computes the eigenvalues and eigenvectors. It then sorts the eigenvectors
        based on their corresponding eigenvalues. To ensure a right-handed coordinate system, the third
        eigenvector is computed as the cross product of the first two.

        Args:
            TM (torch.Tensor): A 3x3 moment tensor matrix.

        Returns:
            torch.Tensor: A 3x3 tensor where each column (along the last dimension) is an eigenvector of the
                        moment tensor. The first two eigenvectors come directly from the sorted decomposition,
                        and the third is the cross product of the first two.
        """

        # Compute the eigenvalues (V) and eigenvectors (S) of the moment tensor.
        V, S = torch.linalg.eigh(TM)

        # Sort the eigenvalues in ascending order and reorder the eigenvectors accordingly.
        inds = torch.argsort(V, dim=-1)
        if len(inds.shape) == 1:
            # For 1D case, simply reorder the eigenvectors.
            S01 = S[..., inds]
        else:
            # For higher dimensions, expand the indices and gather the sorted eigenvectors.
            inds_expanded = inds.unsqueeze(1).expand(-1, S.size(1), -1)
            S01 = torch.gather(S, 2, inds_expanded)

        # Compute the third eigenvector as the cross product of the first two sorted eigenvectors.
        S2 = torch.linalg.cross(S01[..., 0], S01[..., 1])

        # Stack the first eigenvector, the second eigenvector, and the computed third eigenvector
        # to form the final eigenvector representation.
        S_new = torch.stack([S01[..., 0], S01[..., 1], S2], dim=-1)
        return S_new

    @staticmethod
    def ang_from_R1R2(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the angle between two eigenvector matrices.

        The function computes the angle between the eigenvectors of two moment tensors.
        It first multiplies R1 with the transpose of R2, extracts the trace of the resulting
        product, and then scales and clips the value before computing the arccosine to obtain
        the angle.

        Args:
            R1 (torch.Tensor): Eigenvector matrix of the first moment tensor.
            R2 (torch.Tensor): Eigenvector matrix of the second moment tensor.

        Returns:
            torch.Tensor: The angle (in radians) between the eigenvectors.
        """
        # Compute the matrix product of R1 and the transpose of R2 along the last two dimensions.
        matmul = torch.matmul(R1, torch.transpose(R2, -2, -1))

        # Extract the trace of the product by summing over the diagonal elements.
        trace = matmul.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        # Scale the trace to derive the cosine of the angle, and clip to avoid numerical issues.
        # The formula (trace - 1) / 2 originates from the relationship between the trace of a rotation
        # matrix and the rotation angle.
        clip = torch.clip((trace - 1.0) / 2.0, -0.9999999, 0.9999999)

        # Return the arccosine of the clipped value, yielding the angle in radians.
        return torch.arccos(clip)

    @staticmethod
    def compute_conjugate_plane_torch(
        strike: Union[float, torch.Tensor],
        dip: Union[float, torch.Tensor],
        rake: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the conjugate (auxiliary) plane parameters from the given strike, dip, and rake.

        This function calculates the conjugate plane of a focal mechanism by computing the
        adjusted strike, dip, and rake. It converts input angles from degrees to radians, applies the
        slip vector computation, and then converts back to degrees while ensuring the resulting angles
        lie within conventional bounds.

        Args:
            strike (float or torch.Tensor): Strike angle (in degrees) of the first nodal plane.
            dip (float or torch.Tensor): Dip angle (in degrees) of the first nodal plane.
            rake (float or torch.Tensor): Rake angle (in degrees) of the first nodal plane.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Adjusted strike angle (in degrees) for the conjugate plane.
                - Adjusted dip angle (in degrees) for the conjugate plane.
                - Adjusted rake angle (in degrees) for the conjugate plane.
        """
        # Convert input values to torch.Tensor if they are provided as floats.
        if isinstance(strike, float):
            strike = torch.tensor(strike)
        if isinstance(dip, float):
            dip = torch.tensor(dip)
        if isinstance(rake, float):
            rake = torch.tensor(rake)

        # Convert the strike, dip, and rake from degrees to radians.
        strike_rad = torch.deg2rad(strike)
        dip_rad = torch.deg2rad(dip)
        rake_rad = torch.deg2rad(rake)

        # --- Slip Vector Computation ---
        # Compute the numerator ("top") and denominator ("bot") for determining the slip vector direction.
        top = torch.cos(rake_rad) * torch.sin(strike_rad) - torch.cos(
            dip_rad
        ) * torch.sin(rake_rad) * torch.cos(strike_rad)
        bot = torch.cos(rake_rad) * torch.cos(strike_rad) + torch.cos(
            dip_rad
        ) * torch.sin(rake_rad) * torch.sin(strike_rad)

        # Compute the preliminary conjugate strike angle (in degrees) using the arctangent function.
        strike_out = torch.rad2deg(torch.atan2(top, bot))
        # Calculate phi, which is an intermediate angle used in adjusting the plane parameters.
        phi = torch.deg2rad(strike_out - 90)

        # --- Adjustment for Negative Rake ---
        # If the rake is negative, adjust the strike by subtracting 180 degrees.
        strike_out = torch.where(rake_rad < 0, strike_out - 180, strike_out)
        # Normalize the strike to lie within the range [0, 360) degrees.
        strike_out = strike_out % 360

        # --- Compute Conjugate Dip ---
        # The conjugate dip is computed from the product of sin(|rake|) and sin(dip).
        dip_out = torch.rad2deg(
            torch.acos(torch.sin(torch.abs(rake_rad)) * torch.sin(dip_rad))
        )

        # --- Compute Conjugate Rake ---
        # The conjugate rake is derived from the slip vector components adjusted by phi.
        rake_out = -torch.cos(phi) * torch.sin(dip_rad) * torch.sin(
            strike_rad
        ) + torch.sin(phi) * torch.sin(dip_rad) * torch.cos(strike_rad)

        # Clamp the computed value to ensure it lies within the domain of the arccos function.
        rake_out = torch.clamp(rake_out, min=-1.0, max=1.0)

        # Compute the final conjugate rake (in degrees) using arccos, preserving the sign of the original rake.
        rake_out = torch.rad2deg(torch.copysign(torch.acos(rake_out), rake_rad))

        # Return the adjusted conjugate plane parameters:
        # The strike is further adjusted by subtracting 90 and then normalized modulo 360.
        return (strike_out - 90) % 360, dip_out, rake_out


# %%


class MomentMetrics(pl.LightningModule):
    def __init__(
        self,
        metrics: List[str] = [],
        reduction: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MomentMetrics module.

        Args:
            metrics (List[str]): List of metric function names (as strings) to compute.
            reduction (Optional[str]): Reduction mode to apply on metric outputs.
                Options include 'mean', 'max', 'min', 'sum', or 'std'. If None, no reduction is applied.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(MomentMetrics, self).__init__(*args, **kwargs)
        self.metrics = metrics
        self.reduction = reduction
        self.reduction_fn = self.get_reduction(reduction)

    def forward(
        self,
        strike_dip_rake_pred: Union[torch.Tensor, List[float], np.ndarray],
        batch: Union[torch.Tensor, Dict[str, Any], List[float], np.ndarray],
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Compute moment tensor metrics given predicted and ground-truth strike, dip, and rake values.

        Args:
            strike_dip_rake_pred (torch.Tensor or list or np.ndarray):
                Predicted strike, dip, rake values. The last dimension should have 3 elements.
            batch (torch.Tensor or dict or list or np.ndarray):
                Ground-truth strike, dip, rake values. If provided as a dict, it must contain the key "SDR".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, may include a 'device' key to specify the computation device.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping metric names to their computed (and reduced) values.
        """
        # Convert predicted parameters to a torch.Tensor if they are provided as a list or numpy array.
        if isinstance(strike_dip_rake_pred, list | np.ndarray):
            strike_dip_rake_pred = torch.tensor(strike_dip_rake_pred)

        # Convert ground-truth batch to a torch.Tensor.
        if isinstance(batch, list | np.ndarray):
            strike_dip_rake_true = torch.tensor(batch)
        elif isinstance(batch, dict):
            strike_dip_rake_true = batch["SDR"]
        elif isinstance(batch, torch.Tensor):
            strike_dip_rake_true = batch
        else:
            raise ValueError(
                "Batch should be a list, dict with 'SDR' key or torch.Tensor"
            )

        # Ensure that the last dimension contains exactly 3 elements (for strike, dip, and rake).
        assert (
            strike_dip_rake_pred.shape[-1] == 3
        ), "Strike, dip, rake should have 3 values"
        assert (
            strike_dip_rake_true.shape[-1] == 3
        ), "Strike, dip, rake should have 3 values"

        # Move tensors to the specified device if a 'device' keyword argument is provided.
        if "device" in kwargs:
            strike_dip_rake_pred = strike_dip_rake_pred.to(kwargs["device"])
            strike_dip_rake_true = strike_dip_rake_true.to(kwargs["device"])

        # Convert to float64 for improved numerical precision during tensor calculations.
        strike_dip_rake_pred = strike_dip_rake_pred.to(torch.float64)
        strike_dip_rake_true = strike_dip_rake_true.to(torch.float64)

        # Compute the moment tensors from strike, dip, and rake values.
        moment_pred = self.moment_tensor_NED(
            *torch.unbind(strike_dip_rake_pred, dim=-1)
        )
        moment_true = self.moment_tensor_NED(
            *torch.unbind(strike_dip_rake_true, dim=-1)
        )

        # Initialize the output dictionary.
        output: Dict[str, torch.Tensor] = {}

        # Iterate over each metric specified in self.metrics.
        for metric in self.metrics:
            # Dynamically retrieve the metric function using its name.
            metric_value = getattr(self, metric)(moment_pred, moment_true)
            # Apply the reduction function (if any) to the computed metric value.
            output[metric] = self.apply_reduction(metric_value)

        return output

    def get_reduction(
        self, reduction: Optional[str]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Get the reduction function based on the specified reduction mode.

        Args:
            reduction (Optional[str]): Reduction mode as a string.

        Returns:
            Callable: A function that reduces a tensor (e.g., torch.mean, torch.max, etc.)
                    or an identity function if no valid reduction is specified.
        """
        if reduction == "mean":
            return torch.mean
        elif reduction == "max":
            return torch.max
        elif reduction == "min":
            return torch.min
        elif reduction == "sum":
            return torch.sum
        elif reduction == "std":
            return torch.std
        else:
            # If no valid reduction is specified, return the identity function.
            return lambda x: x

    def apply_reduction(
        self, args: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Apply the selected reduction function to the provided tensor(s).

        Args:
            args (torch.Tensor or list/tuple of torch.Tensor):
                The tensor(s) to reduce.

        Returns:
            torch.Tensor or list of torch.Tensor: The reduced tensor(s).
        """
        if isinstance(args, torch.Tensor):
            return self.reduction_fn(args)
        elif isinstance(args, (list, tuple)):
            return [self.reduction_fn(arg) for arg in args]
        else:
            return args

    def DC_CLVD_ISO(
        self, moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the absolute differences between the DC, CLVD, and ISO components of the predicted and true moment tensors.

        The moment tensor is decomposed into three components:
            - DC: Double-couple component.
            - CLVD: Compensated linear vector dipole component.
            - ISO: Isotropic component.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor.
            moment_true (torch.Tensor): Ground-truth moment tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing the absolute differences for the DC, CLVD, and ISO components, respectively.
        """
        # Decompose the predicted moment tensor into its DC, CLVD, and ISO components.
        DC_pred, CLVD_pred, ISO_pred = self.get_DC_CLVD_ISO(moment_pred)
        # Decompose the true moment tensor into its DC, CLVD, and ISO components.
        DC_true, CLVD_true, ISO_true = self.get_DC_CLVD_ISO(moment_true)
        # Return the absolute differences between corresponding components.
        return (
            torch.abs(DC_pred - DC_true),
            torch.abs(CLVD_pred - CLVD_true),
            torch.abs(ISO_pred - ISO_true),
        )

    @staticmethod
    def relative_frobenius(
        moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relative Frobenius norm difference between two moment tensors.

        This function calculates the Frobenius norm of the difference between the true and predicted
        moment tensors and normalizes it by the Frobenius norm of the true moment tensor.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor(s) of shape (..., 3, 3).
            moment_true (torch.Tensor): True moment tensor(s) of shape (..., 3, 3).

        Returns:
            torch.Tensor: A tensor containing the relative Frobenius norm error for each batch element.
        """
        # Compute the Frobenius norm of the difference (error) between the tensors.
        error_norm = torch.norm(moment_true - moment_pred, dim=[-2, -1])
        # Compute the Frobenius norm of the true moment tensor.
        true_norm = torch.norm(moment_true, dim=[-2, -1])
        # Return the relative error.
        return error_norm / true_norm

    @staticmethod
    def relative_eigenvalues(
        moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relative difference between the eigenvalues of predicted and true moment tensors.

        This function first calculates the eigenvalues of both the predicted and true moment tensors
        (assuming symmetry) using torch.linalg.eigvalsh, then computes the norm of the difference between
        these eigenvalues, normalized by the norm of the true eigenvalues.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor(s) of shape (..., 3, 3).
            moment_true (torch.Tensor): True moment tensor(s) of shape (..., 3, 3).

        Returns:
            torch.Tensor: A tensor containing the relative eigenvalue error for each batch element.
        """
        # Compute the eigenvalues of the predicted and true moment tensors.
        eig_pred = torch.linalg.eigvalsh(moment_pred)
        eig_true = torch.linalg.eigvalsh(moment_true)
        # Compute the relative difference between eigenvalues.
        return torch.norm(eig_true - eig_pred, dim=-1) / torch.norm(eig_true, dim=-1)

    @staticmethod
    def relative_eigenvectors(
        moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relative difference between the eigenvector decompositions of two moment tensors.

        This function calculates the eigenvalue decomposition (eigenvalues and eigenvectors) for both
        predicted and true moment tensors using torch.linalg.eigh, then computes the norm of the difference
        between these decompositions normalized by the norm of the true eigen decomposition.

        Note: The comparison of eigenvectors is sensitive to sign ambiguities and ordering; this metric
        assumes a consistent sorting of eigenvectors.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor(s) of shape (..., 3, 3).
            moment_true (torch.Tensor): True moment tensor(s) of shape (..., 3, 3).

        Returns:
            torch.Tensor: A tensor containing the relative error between the eigen decompositions for each batch element.
        """
        # Compute eigen-decompositions (eigenvalues and eigenvectors) for the predicted and true tensors.
        eig_pred = torch.linalg.eigh(moment_pred)
        eig_true = torch.linalg.eigh(moment_true)
        # Compute the relative difference between the eigen decompositions.
        return torch.norm(eig_true - eig_pred, dim=-1) / torch.norm(eig_true, dim=-1)

    @staticmethod
    def angular_distance(
        moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the angular distance between two moment tensors.

        This function computes the angular distance by evaluating the arccosine of the normalized
        trace of the product of the transposed true moment tensor and the predicted moment tensor.
        The result is clamped to the valid range of [-1, 1] to ensure numerical stability.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor(s) of shape (..., 3, 3).
            moment_true (torch.Tensor): True moment tensor(s) of shape (..., 3, 3).

        Returns:
            torch.Tensor: A tensor containing the angular distance (in radians) for each batch element.
        """
        # Compute the product of the transposed true tensor and the predicted tensor.
        prod = torch.matmul(torch.transpose(moment_true, -2, -1), moment_pred)
        # Extract the sum of diagonal elements (trace) from the product.
        trace = prod.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        # Compute the product of the Frobenius norms of both moment tensors.
        norm_product = torch.norm(moment_true, dim=[-2, -1]) * torch.norm(
            moment_pred, dim=[-2, -1]
        )
        # Compute the normalized trace and clamp its values between -1 and 1.
        normalized_trace = torch.clamp(trace / norm_product, min=-1.0, max=1.0)
        # Return the angular distance by computing the arccosine of the normalized trace.
        return torch.acos(normalized_trace)

    @staticmethod
    def get_DC_CLVD_ISO(
        moment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose a moment tensor into its DC, CLVD, and ISO components.

        This function first calculates the isotropic component (M_iso) by taking the average of the diagonal
        elements. It then computes the deviatoric part of the moment tensor and its eigenvalues. Using the
        sorted eigenvalues, the function computes:
            - DC (Double Couple) component,
            - CLVD (Compensated Linear Vector Dipole) component,
            - ISO (Isotropic) component.

        Args:
            moment (torch.Tensor): Moment tensor(s) of shape (Batch, 3, 3).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - DC: Tensor of shape (Batch,) representing the double couple component.
                - CLVD: Tensor of shape (Batch,) representing the CLVD component.
                - ISO: Tensor of shape (Batch,) representing the isotropic component.
        """
        # Compute the isotropic moment by averaging the diagonal elements.
        m_iso = moment.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) / 3.0  # (Batch,)
        # Create an identity matrix and repeat it for each element in the batch.
        eye = torch.eye(3).unsqueeze(0).repeat(moment.size(0), 1, 1).to(moment.device)
        # Construct the isotropic part of the moment tensor.
        M_iso = eye * m_iso.view(-1, 1, 1)  # (Batch, 3, 3)
        # Compute the deviatoric part of the moment tensor.
        M_dev = moment - M_iso  # (Batch, 3, 3)
        # Compute the eigenvalues of the deviatoric tensor.
        eigvals = torch.linalg.eigvalsh(M_dev)  # (Batch, 3)
        # Sort the eigenvalues in ascending order.
        eigvals = torch.sort(eigvals, dim=-1).values  # (Batch, 3)
        # Define a normalization factor based on the absolute extremes of the eigenvalues.
        M0 = 0.5 * ((eigvals[..., 2]).abs() + (eigvals[..., 0]).abs())  # (Batch,)
        # Compute the Double Couple (DC) component as the normalized difference between the max and min eigenvalues.
        DC = (eigvals[..., 2] - eigvals[..., 0]).abs() / (2 * M0)  # (Batch,)
        # Compute the CLVD component as the normalized sum of the absolute values of the max and min eigenvalues.
        CLDV = (eigvals[..., 2] + eigvals[..., 0]).abs() / (2 * M0)  # (Batch,)
        # Compute the ISO component as the normalized absolute isotropic moment.
        ISO = m_iso.abs() / M0  # (Batch,)
        return DC, CLDV, ISO

    @staticmethod
    def plane_to_torchtensor(
        strike: Union[float, torch.Tensor],
        dip: Union[float, torch.Tensor],
        rake: Union[float, torch.Tensor],
        mag: Union[float, torch.Tensor] = 2.0,
    ) -> torch.Tensor:
        """
        Convert strike, dip, and rake angles (in degrees) to a moment tensor representation.

        This function takes the strike, dip, and rake angles corresponding to the first nodal plane
        of an earthquake source and computes the 3x3 moment tensor. The seismic moment is derived from
        the magnitude using a scaling relation, and the tensor components are computed using standard
        trigonometric formulas. Finally, the moment tensor is scaled from dyne-cm to N-m.

        Args:
            strike (float or torch.Tensor): Strike angle (in degrees) from the first nodal plane.
            dip (float or torch.Tensor): Dip angle (in degrees) from the first nodal plane.
            rake (float or torch.Tensor): Rake angle (in degrees) from the first nodal plane.
            mag (float or torch.Tensor, optional): Earthquake magnitude used to compute the seismic moment.
                                                Default is 2.0.

        Returns:
            torch.Tensor: A 3x3 tensor representing the moment tensor with components arranged as:
                        [[mrr, mrt, mrp],
                        [mrt, mtt, mtp],
                        [mrp, mtp, mpp]]
        """

        # If the inputs are provided as floats, convert them to torch tensors with gradient tracking enabled.
        if isinstance(strike, float):
            strike = torch.tensor(strike, requires_grad=True)
        if isinstance(dip, float):
            dip = torch.tensor(dip, requires_grad=True)
        if isinstance(rake, float):
            rake = torch.tensor(rake, requires_grad=True)
        if isinstance(mag, float):
            mag = torch.tensor(mag, requires_grad=True)

        # Define conversion factor from degrees to radians.
        d2r = torch.tensor(torch.pi / 180.0)

        # Compute the seismic moment:
        # The scaling relation is given by: magpow = mag * 1.5 + 16.1, so the moment is 10^(magpow).
        magpow = mag * 1.5 + 16.1
        mom = torch.pow(10, magpow)

        # Compute trigonometric functions for the given angles (converted to radians).
        sin_2_dip_d2r = torch.sin(2 * dip * d2r)  # sin(2*dip)
        sin_rake_d2r = torch.sin(rake * d2r)  # sin(rake)
        sin_2_strike_d2r = torch.sin(2 * strike * d2r)  # sin(2*strike)
        sin_strike_d2r = torch.sin(strike * d2r)  # sin(strike)
        cos_strike_d2r = torch.cos(strike * d2r)  # cos(strike)
        cos_2_dip_d2r = torch.cos(2 * dip * d2r)  # cos(2*dip)
        sin_dip_d2r_cos_rake_d2r = torch.sin(dip * d2r) * torch.cos(
            rake * d2r
        )  # sin(dip)*cos(rake)
        cos_dip_d2r_cos_rake_d2r = torch.cos(dip * d2r) * torch.cos(
            rake * d2r
        )  # cos(dip)*cos(rake)

        # Calculate moment tensor components based on the standard formulas.
        mrr = mom * sin_2_dip_d2r * sin_rake_d2r
        mtt = -mom * (
            (sin_dip_d2r_cos_rake_d2r * sin_2_strike_d2r)
            + (sin_2_dip_d2r * sin_rake_d2r * (sin_strike_d2r * sin_strike_d2r))
        )
        mpp = mom * (
            (sin_dip_d2r_cos_rake_d2r * sin_2_strike_d2r)
            - (sin_2_dip_d2r * sin_rake_d2r * (cos_strike_d2r * cos_strike_d2r))
        )
        mrt = -mom * (
            (cos_dip_d2r_cos_rake_d2r * cos_strike_d2r)
            + (cos_2_dip_d2r * sin_rake_d2r * sin_strike_d2r)
        )
        mrp = mom * (
            (cos_dip_d2r_cos_rake_d2r * sin_strike_d2r)
            - (cos_2_dip_d2r * sin_rake_d2r * cos_strike_d2r)
        )
        mtp = -mom * (
            (sin_dip_d2r_cos_rake_d2r * torch.cos(2 * strike * d2r))
            + (0.5 * sin_2_dip_d2r * sin_rake_d2r * sin_2_strike_d2r)
        )

        # Assemble the moment tensor matrix by stacking the computed components.
        # The resulting matrix has the structure:
        # [[mrr, mrt, mrp],
        #  [mrt, mtt, mtp],
        #  [mrp, mtp, mpp]]
        mt_matrix = torch.stack(
            [
                torch.stack([mrr, mrt, mrp], dim=-1),
                torch.stack([mrt, mtt, mtp], dim=-1),
                torch.stack([mrp, mtp, mpp], dim=-1),
            ],
            dim=-1,
        )

        # Convert the units from dyne-cm to N-m (scaling factor of 1e-7).
        mt_matrix = mt_matrix * 1e-7
        return mt_matrix

    @staticmethod
    def cosine_similarity(
        moment_pred: torch.Tensor, moment_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cosine similarity between two moment tensors.

        The moment tensors are first flattened (starting from dimension 1) so that each sample
        becomes a one-dimensional vector. Then, the cosine similarity is computed along the last dimension.

        Args:
            moment_pred (torch.Tensor): Predicted moment tensor of shape (..., m, n).
            moment_true (torch.Tensor): True moment tensor of shape (..., m, n).

        Returns:
            torch.Tensor: A tensor containing cosine similarity values for each batch element.
        """
        # Flatten the tensors from dimension 1 onward and compute cosine similarity.
        return F.cosine_similarity(
            moment_pred.flatten(start_dim=1), moment_true.flatten(start_dim=1), dim=-1
        )

    @staticmethod
    def moment_tensor_NED(
        strike: Union[float, torch.Tensor],
        dip: Union[float, torch.Tensor],
        rake: Union[float, torch.Tensor],
        M0: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        Construct the moment tensor in the North-East-Down (NED) coordinate system from strike, dip, and rake angles.

        This function converts strike, dip, and rake angles from degrees to radians and computes the moment tensor
        using the following relation:
            M = M0 [ d n^T + n d^T ] = 2 * M0 * sym(d n^T)
        where:
            - 'n' is the unit normal vector to the fault plane.
            - 'd' is the slip vector.
        The tensor M is symmetric and is scaled by the scalar moment M0.

        Args:
            strike (float or torch.Tensor): Strike angle (in degrees).
            dip (float or torch.Tensor): Dip angle (in degrees).
            rake (float or torch.Tensor): Rake angle (in degrees).
            M0 (float or torch.Tensor, optional): Scalar moment magnitude. Default is 1.0.

        Returns:
            torch.Tensor: A 3x3 moment tensor in NED coordinates.
        """
        # Convert input angles to torch.Tensor if they are provided as floats.
        if isinstance(strike, float):
            strike = torch.tensor(strike)
        if isinstance(dip, float):
            dip = torch.tensor(dip)
        if isinstance(rake, float):
            rake = torch.tensor(rake)

        # Define conversion factor from degrees to radians.
        d2r = torch.tensor(torch.pi / 180.0)

        # Convert angles from degrees to radians.
        phi = strike * d2r  # Strike in radians.
        delta = dip * d2r  # Dip in radians.
        lam = rake * d2r  # Rake in radians.

        # Compute the unit normal vector 'n' in NED coordinates.
        # n_N (North), n_E (East), and n_D (Down) components.
        n = torch.stack(
            [
                torch.sin(delta) * torch.sin(phi),  # n_N
                torch.sin(delta) * torch.cos(phi),  # n_E
                torch.cos(delta),  # n_D
            ],
            dim=-1,
        )

        # Compute the slip vector 'd' in NED coordinates.
        d = torch.stack(
            [
                # d_N: Combination of strike, dip, and rake effects.
                torch.cos(lam) * torch.cos(delta) * torch.sin(phi)
                - torch.sin(lam) * torch.cos(phi),
                # d_E: Combination of strike, dip, and rake effects.
                torch.cos(lam) * torch.cos(delta) * torch.cos(phi)
                + torch.sin(lam) * torch.sin(phi),
                # d_D: Down component (with a negative sign to indicate downward direction).
                -torch.cos(lam) * torch.sin(delta),
            ],
            dim=-1,
        )

        # Construct the symmetric moment tensor M using the formula:
        # M = M0 * [d * n^T + n * d^T]
        # Use unsqueeze to align dimensions for proper matrix multiplication.
        M = M0 * (
            torch.matmul(d.unsqueeze(-1), n.unsqueeze(-2))
            + torch.matmul(n.unsqueeze(-1), d.unsqueeze(-2))
        )

        return M


# %%


class FocalMetrics(pl.LightningModule):
    def __init__(
        self,
        measure: Union[str, List[str]] = "mse",
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the FocalMetrics module for computing focal mechanism error metrics.

        Args:
            measure (str or List[str], optional):
                The error metric(s) to use. Supported values are "mse" (mean squared error)
                and/or "mae" (mean absolute error). If a single string is provided, it will be converted
                to a list.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(FocalMetrics, self).__init__()
        # Convert measure to a list if a single string is provided.
        if isinstance(measure, str):
            measure = [measure]
        # Ensure that each provided measure is either "mse" or "mae".
        assert np.all(
            [m in ["mse", "mae"] for m in measure]
        ), "Measure should be 'mse' or 'mae' or both"
        self.measure = measure
        # Define a normalization lambda that scales strike, dip, and rake values.
        # Normalization is performed as follows:
        #   - Strike is divided by 360 (full circle)
        #   - Dip is divided by 90 (maximum dip)
        #   - Rake is shifted by +180 and divided by 360.
        self.normalizator = lambda x: torch.stack(
            [(x[:, 0] / 360.0), (x[:, 1] / 90.0), (x[:, 2] + 180.0) / 360.0], dim=1
        )

    def forward(
        self,
        strike_dip_rake_pred: torch.Tensor,
        batch: Union[torch.Tensor, Dict[str, Any]],
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the focal mechanism metrics between predicted and true strike, dip, and rake values.

        Args:
            strike_dip_rake_pred (torch.Tensor):
                Predicted strike, dip, and rake values with shape [batch, 3].
            batch (torch.Tensor or dict):
                Ground-truth strike, dip, and rake values. If provided as a dict, it must contain the key "SDR".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Optionally, a 'device' key may be provided to move the data to a specific device.

        Returns:
            Dict[str, torch.Tensor]:
                A dictionary with keys corresponding to each error metric (e.g., "mse", "mae") and values containing
                the computed metric (cast to float32).
        """
        # Ensure the predicted tensor is of shape [batch, 3] and is a torch.Tensor.
        assert (
            isinstance(strike_dip_rake_pred, torch.Tensor)
            and strike_dip_rake_pred.shape[-1] == 3
            and len(strike_dip_rake_pred.shape) == 2
        ), "Strike, dip, rake should have 3 values"

        # Extract ground-truth values from the batch.
        if isinstance(batch, dict):
            strike_dip_rake_true = batch["SDR"]
        elif isinstance(batch, torch.Tensor):
            strike_dip_rake_true = batch
        else:
            raise ValueError("Batch should be a dict with 'SDR' key or torch.Tensor")

        # Ensure ground-truth tensor is of shape [batch, 3].
        assert (
            strike_dip_rake_true.shape[-1] == 3 and len(strike_dip_rake_true.shape) == 2
        ), "Strike, dip, rake should have dimension [batch, 3]"

        # If a device is specified in kwargs, move tensors to that device.
        if "device" in kwargs:
            strike_dip_rake_pred = strike_dip_rake_pred.to(kwargs["device"])
            strike_dip_rake_true = strike_dip_rake_true.to(kwargs["device"])

        # Expand the predicted tensor to have an extra dimension for two conjugate solutions.
        # After unsqueeze, shape becomes [batch, 3, 1] and then expands to [batch, 3, 2].
        strike_dip_rake_pred = strike_dip_rake_pred.unsqueeze(-1).expand(
            *strike_dip_rake_pred.shape, 2
        )

        # Convert tensors to float64 for numerical precision.
        strike_dip_rake_pred = strike_dip_rake_pred.to(torch.float64)
        strike_dip_rake_true = strike_dip_rake_true.to(torch.float64)

        # Compute the true pair of focal mechanism parameters: original and its conjugate.
        # Note: The function 'conjugate_torch' is assumed to be defined elsewhere.
        strike_dip_rake_couple = torch.stack(
            [strike_dip_rake_true, conjugate_torch(strike_dip_rake_true)], dim=-1
        )

        # Apply normalization to both predicted and true focal mechanism parameters.
        strike_dip_rake_pred_scaled = self.normalizator(strike_dip_rake_pred)
        strike_dip_rake_couple_scaled = self.normalizator(strike_dip_rake_couple)

        output = {}
        # Compute mean squared error if "mse" is in the measure list.
        if "mse" in self.measure:
            # Calculate squared differences.
            squared_diff = torch.pow(strike_dip_rake_pred_scaled - strike_dip_rake_couple_scaled, 2)
            # Average error per parameter.
            mean_error = torch.mean(squared_diff, dim=1)
            # Take the minimum error across the two conjugate solutions.
            min_error = torch.min(mean_error, dim=-1, keepdim=False).values
            # Average over the batch and cast to float32.
            output["mse"] = torch.mean(min_error).to(torch.float32)

        # Compute mean absolute error if "mae" is in the measure list.
        if "mae" in self.measure:
            # Calculate absolute differences.
            abs_diff = torch.abs(strike_dip_rake_pred_scaled - strike_dip_rake_couple_scaled)
            # Average error per parameter.
            mean_abs_error = torch.mean(abs_diff, dim=1)
            # Take the minimum error across the two conjugate solutions.
            min_abs_error = torch.min(mean_abs_error, dim=-1, keepdim=False).values
            # Average over the batch and cast to float32.
            output["mae"] = torch.mean(min_abs_error).to(torch.float32)

        return output


# %%


class StereoDiscrepancyLoss(pl.LightningModule):
    def __init__(self, reduction="mean", normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        else:
            self.reduction = lambda x: x

        self.normalize = normalize

    def forward(self, strike_dip_rake_pred, batch, *args, **kwargs):

        if "device" in kwargs:
            self.to(kwargs["device"])
            strike_dip_rake_pred = strike_dip_rake_pred.to(self.device)

        strike_conj, dip_conj, _ = self.compute_conjugate_plane_torch(
            *strike_dip_rake_pred.split(1, dim=1)
        )

        strike, dip, _ = strike_dip_rake_pred.split(1, dim=1)

        assert isinstance(batch, dict), "Batch should be a dictionary"
        assert "az" in batch, "Batch should contain the azimuths"
        assert "ih" in batch, "Batch should contain the takeoff angles"
        assert "Polarities" in batch, "Batch should contain the Polarities"

        azimuths = batch["az"].to(self.device)
        takeoff_angles = batch["ih"].to(self.device)
        pol = batch["Polarities"].to(self.device)

        x_staz, y_staz = self.convert_azimuth_takeoffs_to_xy_torch(
            azimuths, takeoff_angles
        )
        xy_staz = torch.stack([x_staz, y_staz], dim=-1)

        x_plane1, y_plane1 = self.proj_plane(strike, dip)
        circ1 = torch.stack([x_plane1, y_plane1], dim=-1)
        circ1_staz = circ1.unsqueeze(1).repeat(1, xy_staz.shape[1], 1, 1)

        x_plane2, y_plane2 = self.proj_plane(strike_conj, dip_conj)
        circ2 = torch.stack([x_plane2, y_plane2], dim=-1)
        circ2_staz = circ2.unsqueeze(1).repeat(1, xy_staz.shape[1], 1, 1)

        in_out_circ1 = self.winding_number_torch(circ1_staz, xy_staz)
        in_out_circ2 = self.winding_number_torch(circ2_staz, xy_staz)

        in_out = (in_out_circ1 + in_out_circ2) % 2

        P, T = FocalMechanism_torch(strike_dip_rake_pred).get_PT()

        xPT, yPT = self.convert_azimuth_takeoffs_to_xy_torch(
            torch.stack([P[:, 1], T[:, 1]], dim=-1),
            torch.stack([P[:, 0], T[:, 0]], dim=-1),
        )
        xy_PT = torch.stack([xPT, yPT], dim=-1)

        # Compute the winding number for the P and T points
        circ1_PT = circ1.unsqueeze(1).repeat(1, xy_PT.shape[1], 1, 1)
        circ2_PT = circ2.unsqueeze(1).repeat(1, xy_PT.shape[1], 1, 1)
        in_out1_PT = self.winding_number_torch(circ1_PT, xy_PT)
        in_out2_PT = self.winding_number_torch(circ2_PT, xy_PT)
        in_out_PT = (in_out1_PT + in_out2_PT) % 2

        # Compute the discrepancy error
        areaP = self.equal_torch(in_out, in_out_PT[0, 0]) * self.equal_torch(
            pol, torch.tensor(1.0)
        )

        areaT = self.equal_torch(in_out, in_out_PT[0, 1]) * self.equal_torch(
            pol, torch.tensor(-1.0)
        )

        output = areaP.sum(-1) + areaT.sum(-1)

        if self.normalize:
            output = output / pol.abs().sum(-1)

        return self.reduction(output)

    @staticmethod
    def compute_conjugate_plane_torch(strike, dip, rake):
        # Convert to radians
        strike_rad = torch.deg2rad(strike)
        dip_rad = torch.deg2rad(dip)
        rake_rad = torch.deg2rad(rake)

        # Slip Vector Computation
        top = torch.cos(rake_rad) * torch.sin(strike_rad) - torch.cos(
            dip_rad
        ) * torch.sin(rake_rad) * torch.cos(strike_rad)
        bot = torch.cos(rake_rad) * torch.cos(strike_rad) + torch.cos(
            dip_rad
        ) * torch.sin(rake_rad) * torch.sin(strike_rad)

        strike_out = torch.rad2deg(torch.atan2(top, bot))
        phi = torch.deg2rad(strike_out - 90)

        # Adjust strike_out if rake is negative
        strike_out = torch.where(rake_rad < 0, strike_out - 180, strike_out)
        strike_out = strike_out % 360

        # Compute dip_out
        dip_out = torch.rad2deg(
            torch.acos(torch.sin(torch.abs(rake_rad)) * torch.sin(dip_rad))
        )

        # Compute rake_out
        rake_out = -torch.cos(phi) * torch.sin(dip_rad) * torch.sin(
            strike_rad
        ) + torch.sin(phi) * torch.sin(dip_rad) * torch.cos(strike_rad)

        # Clamp rake_out between -1 and 1 if necessary
        rake_out = torch.clamp(rake_out, min=-1.0, max=1.0)

        # Calculate final rake_out with correct sign
        rake_out = torch.rad2deg(torch.copysign(torch.acos(rake_out), rake_rad))

        return (strike_out - 90) % 360, dip_out, rake_out

    @staticmethod
    def convert_azimuth_takeoffs_to_xy_torch(azimuth, takeoff):
        """
        Calcola la posizione di una stazione usando la proiezione equal-area con PyTorch.

        Parametri:
        - azimuth: angolo in gradi (torch tensor o float)
        - takeoff: angolo in gradi (torch tensor o float)

        Ritorna:
        - x, y: coordinate in proiezione equal-area (torch tensor)
        """

        # Converti azimut e takeoff da gradi a radianti
        trend_rad = torch.deg2rad(azimuth)
        plunge_rad = torch.deg2rad((90 - takeoff) % 180)

        # Calcola la colatitudine
        colatitude = torch.pi / 2 - plunge_rad

        # Proiezione Equal-Area
        r = torch.sqrt(torch.tensor(2.0)) * torch.sin(colatitude / 2)
        x = r * torch.sin(trend_rad)
        y = r * torch.cos(trend_rad)

        return x, y

    def proj_plane(self, strike, dip):
        """
        Projects the plane on the Stereonet.
        """

        # Convert the strike and dip to radians
        strike_rad = torch.deg2rad(strike)
        dip_rad = torch.deg2rad(dip)

        azimuths = torch.linspace(0, 360, steps=361).to(self.device)
        azimuths_rad = torch.deg2rad(azimuths)
        azimuths_rad = azimuths_rad.unsqueeze(0).repeat(strike_rad.shape[0], 1)

        apparent_dips = torch.atan(
            torch.tan(dip_rad) * torch.sin(azimuths_rad - strike_rad)
        )

        colatitudes = torch.pi / 2 - apparent_dips

        r = torch.sqrt(torch.tensor(2.0)) * torch.sin(colatitudes / 2)
        x_proj = r * torch.sin(azimuths_rad)
        y_proj = r * torch.cos(azimuths_rad)

        return x_proj, y_proj

    def winding_number_torch(self, polygon, point):
        """
        Calcola il winding number di un batch di punti rispetto a un batch di poligoni in modo vettorizzato con PyTorch.

        Parametri:
        polygon (torch.Tensor): Un tensore (B, S, N, 2) che rappresenta un batch di poligoni con N vertici.
        point (torch.Tensor): Un tensore (B, S, 2) che rappresenta un batch di punti per stazione da testare.

        Ritorna:
        torch.Tensor: Un tensore (B,) contenente il winding number per ogni punto rispetto al rispettivo poligono.
        """

        # Aggiungiamo una dimensione extra per il batch di punti così possiamo fare broadcasting
        point = point.unsqueeze(2)  # Ora ha forma (B, S, 1, 2)

        # Otteniamo i vertici del poligono e quelli rotolati per formare gli spigoli
        p0 = polygon
        p1 = torch.roll(polygon, shifts=-1, dims=2)  # Rotola i vertici di una posizione

        # Condizioni di attraversamento verso l'alto e verso il basso
        upward_crossing = self.less_equal_torch(
            p0[..., 1], point[..., 1]
        ) * self.greater_torch(p1[..., 1], point[..., 1])
        downward_crossing = self.greater_torch(
            p0[..., 1], point[..., 1]
        ) * self.less_equal_torch(p1[..., 1], point[..., 1])

        # Calcola se i punti sono a sinistra di ogni spigolo del batch
        is_left = self.is_left_torch(p0, p1, point)

        second_terms = (
            downward_crossing * self.less_torch(is_left, torch.tensor(0.0))
        ) * -1

        cond = upward_crossing * (self.greater_torch(is_left, torch.tensor(0.0)))
        winding_contributions = cond + (second_terms * (1 - cond))

        # Somma lungo la dimensione dei vertici per ottenere il winding number finale per ogni batch
        winding_number = torch.sum(winding_contributions, dim=2)

        # return winding_number
        return winding_number.abs()

    def equal_torch(self, a, b):
        return self.less_equal_torch(a, b) * self.greater_equal_torch(a, b)

    @staticmethod
    def is_left_torch(p0, p1, p2):
        """Calcola il determinante batch-wise per capire se p2 è a sinistra della linea (p0, p1)"""
        return (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1]) - (
            p2[..., 0] - p0[..., 0]
        ) * (p1[..., 1] - p0[..., 1])

    @staticmethod
    def less_torch(a, b):
        return torch.clamp(torch.sign(b - a), 0, 1)

    @staticmethod
    def less_equal_torch(a, b):
        e = torch.finfo().eps
        return torch.clamp(torch.sign(b - a + e), 0, 1)

    @staticmethod
    def greater_torch(a, b):
        return torch.clamp(torch.sign(a - b), 0, 1)

    @staticmethod
    def greater_equal_torch(a, b):
        e = torch.finfo().eps
        return torch.clamp(torch.sign(a - b + e), 0, 1)
    

# %%
