#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%

from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import List, Dict, Union

# %%

def _beta_distribution(k: float, mode: float = 0.5) -> torch.distributions.Beta:
    """
    Calculates the alpha and beta parameters for a Beta distribution.

    Args:
        k (float): A parameter related to the total concentration.
        mode (float, optional): The mode of the Beta distribution. Defaults to 0.5.

    Returns:
        torch.distributions.Beta: A Beta distribution object.
    """
    # Calculates the alpha and beta parameters of the Beta distribution
    # The alpha parameter is calculated based on the mode and total concentration k.
    alpha = mode * (k - 2) + 1
    # The beta parameter is calculated based on the mode and total concentration k.
    beta = (1 - mode) * (k - 2) + 1
    # Create a Beta distribution object using the calculated alpha and beta.
    dist = torch.distributions.Beta(alpha, beta)
    return dist

class MinMaxScalerLayer(pl.LightningModule):
    """
    A PyTorch Lightning module that performs Min-Max scaling on input tensors.

    This layer scales input data X from an original range [boundaries_min, boundaries_max]
    to a new specified range [scaled_min, scaled_max].
    """
    def __init__(
        self,
        boundary: Union[List[float], None] = None,
        scaling_range: Union[List[float], None] = None,
    ):
        """
        Initializes the MinMaxScalerLayer.

        Args:
            boundary (list, optional): A list defining the original minimum and maximum
                                        values of the data. Can be [min, max] for a single
                                        range, or [min1, max1, min2, max2, ...] for multiple
                                        ranges corresponding to different features.
                                        Defaults to None, which sets boundaries to [0, 1].
            scaling_range (list, optional): A list defining the target minimum and maximum
                                            values for scaling. Can be [min, max] for a single
                                            range, or [min1, max1, min2, max2, ...] for multiple
                                            ranges. Defaults to None, which sets the scaling
                                            range to [0, 1].
        """
        super(MinMaxScalerLayer, self).__init__()

        # Store the desired scaling range for later use in the forward pass.
        self.scaling_range = scaling_range

        # Check if boundary values are provided.
        if boundary is not None:
            # Ensure that the boundary input is a list.
            assert isinstance(boundary, list), "Boundary must be a list"
            # Check if multiple boundaries are provided (e.g., for multi-feature scaling).
            if len(boundary) > 2:
                # Extract minimum boundaries (every second element starting from the first).
                boundaries_min = torch.tensor(boundary[::2]).unsqueeze(0)
                # Extract maximum boundaries (every second element starting from the second).
                boundaries_max = torch.tensor(boundary[1::2]).unsqueeze(0)
                # If a scaling_range is provided, use it.
                if scaling_range is not None:
                    # Extract target minimums for scaling.
                    minumums = torch.tensor(scaling_range[::2]).unsqueeze(0)
                    # Extract target maximums for scaling.
                    maximums = torch.tensor(scaling_range[1::2]).unsqueeze(0)
                else:
                    # Default target scaling range to [0, 1] if not specified.
                    minumums = torch.tensor(0.0)
                    maximums = torch.tensor(1.0)
            else:
                # Handle the case of a single boundary pair for all features.
                boundaries_min = torch.tensor(boundary[0])
                boundaries_max = torch.tensor(boundary[1])
                # If a scaling_range is provided, use it.
                if scaling_range is not None:
                    # Extract target minimum for scaling.
                    minumums = torch.tensor(scaling_range[0])
                    # Extract target maximum for scaling.
                    maximums = torch.tensor(scaling_range[1])
                else:
                    # Default target scaling range to [0, 1] if not specified.
                    minumums = torch.tensor(0.0)
                    maximums = torch.tensor(1.0)

            # Register non-trainable tensors as buffers so they are moved to the correct device.
            self.register_buffer(
                "scaled_min", minumums
            )  # Registering as buffer to avoid updating during training
            self.register_buffer("scaled_max", maximums)
            self.register_buffer("boundaries_min", boundaries_min)
            self.register_buffer("boundaries_max", boundaries_max)
        else:
            # If no boundary is provided, default all boundaries and scaling range to [0, 1].
            self.register_buffer("boundaries_min", torch.tensor(0.0))
            self.register_buffer("boundaries_max", torch.tensor(1.0))
            self.register_buffer("scaled_min", torch.tensor(0.0))
            self.register_buffer("scaled_max", torch.tensor(1.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the Min-Max scaling.

        The formula used is:
        X_scaled = (X - X_min_original) / (X_max_original - X_min_original) * (X_max_target - X_min_target) + X_min_target

        Args:
            X (torch.Tensor): The input tensor to be scaled.

        Returns:
            torch.Tensor: The scaled tensor.
        """
        # Normalize X to a [0, 1] range based on original boundaries
        X_std = (X - self.boundaries_min) / (self.boundaries_max - self.boundaries_min)
        # Scale the normalized X to the target scaling_range
        return X_std * (self.scaled_max - self.scaled_min) + self.scaled_min


class ScaledSigmoid(pl.LightningModule):
    """
    A PyTorch Lightning module that implements a scaled sigmoid activation function.

    The sigmoid function is scaled by a factor 'alpha'.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ScaledSigmoid layer.

        Args:
            alpha (float, optional): The scaling factor for the sigmoid function. Defaults to 1.0.
        """
        super(ScaledSigmoid, self).__init__()
        # Store the alpha parameter which controls the steepness of the sigmoid curve.
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the scaled sigmoid function to the input tensor.

        The formula used is:
        f(x) = 1 / (1 + exp(-alpha * x))

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the scaled sigmoid.
        """
        # Apply the scaled sigmoid activation function.
        return 1 / (1 + torch.exp(-self.alpha * x))


# %%


class AmplitudePolaritiesModel(pl.LightningModule):
    """
    A PyTorch Lightning module for predicting strike, dip, and rake angles
    based on amplitudes and polarities from seismic stations, incorporating
    station coordinates and multi-head attention.
    """
    def __init__(
        self,
        n_stations: int,
        xyz_boundary: Union[List[float], None],
        scaling_range: Union[List[float], None] = None,
        generator: Union[torch.Generator, None] = None,
    ):
        """
        Initializes the AmplitudePolaritiesModel.

        Args:
            n_stations (int): The number of seismic stations.
            xyz_boundary (Union[List[float], None], optional): Boundary for XYZ coordinates for scaling.
                                                                Expected to be [min_x, max_x, min_y, max_y, min_z, max_z].
                                                                Defaults to None.
            scaling_range (Union[List[float], None], optional): Target scaling range for the MinMaxScalerLayers.
                                                                    Expected to be [min_target, max_target] for each dimension.
                                                                    Defaults to None.
            generator (Union[torch.Generator, None], optional): A PyTorch random number generator. Defaults to None.
        """
        super().__init__()
        self.n_stations = n_stations
        
        self.xyz_boundary = xyz_boundary

        # Initialize MinMaxScalerLayer for XYZ coordinates.
        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        # Initialize MinMaxScalerLayer for XY coordinates (first 4 elements of xyz_boundary).
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6  # Represents 2 components for strike, 2 for dip, 2 for rake (sin/cos representation)

        # Fully connected layers for XYZ station embeddings
        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.relu10xyz = torch.nn.ReLU()

        # Convolutional layers for Amplitude processing with different dilations
        self.Aconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )
        # Convolutional layers for Polarity processing with different dilations
        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        # Flatten layers for convolutional outputs
        self.flattenA = nn.Flatten()
        self.flattenP = nn.Flatten()
        
        # Initial fully connected layer for combined polarity features
        self.fc1cp0 = torch.nn.LazyLinear(1024)
        self.relu1cp0 = torch.nn.ReLU()

        # Multi-head attention layers for Amplitude and Polarity features
        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )
        self.multihead_attention2 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )

        # Fully connected layers for combined polarity features after attention
        self.fc1cp = torch.nn.Linear(1024, 512)
        self.relu1cp = torch.nn.ReLU()
        self.fc2cp = torch.nn.Linear(512, 256)
        self.relu2cp = torch.nn.ReLU()
        self.fc3cp = torch.nn.Linear(256, 128)
        self.relu3cp = torch.nn.ReLU()
        self.fc4cp = torch.nn.Linear(128, 64)
        self.relu4cp = torch.nn.ReLU()
        self.fc5cp = torch.nn.Linear(64, 32)
        self.relu5cp = torch.nn.ReLU()
        self.fc6cp = torch.nn.Linear(32, 16)
        self.relu6cp = torch.nn.ReLU()
        self.fc7cp = torch.nn.Linear(16, 8)
        self.relu7cp = torch.nn.ReLU()
        self.fc8cp = torch.nn.Linear(8, self.output_shape)
        self.relu8cp = torch.nn.Tanh() # Activation for the final combined features

        # Output layers for strike angle prediction
        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh() # Tanh for strike (sin/cos components typically range -1 to 1)
        
        # Output layers for dip angle prediction
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0) # Scaled Sigmoid for dip

        # Output layers for rake angle prediction
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation1 = torch.nn.Tanh() # Tanh for sin(rake)
        self.fc_rake_activation2 = ScaledSigmoid(alpha=3.0) # Scaled Sigmoid for cos(rake)

        # Fully connected layers for Amplitude features, potentially used for gating
        self.fc00A = torch.nn.LazyLinear(1024)
        self.relu00A = torch.nn.ReLU()
        self.fc0A = torch.nn.Linear(1024, 256)
        self.relu0A = torch.nn.ReLU()
        self.fc1A = torch.nn.Linear(256, 64)
        self.relu1A = torch.nn.ReLU()
        self.fc2A = torch.nn.Linear(64, 8)
        self.relu2A = torch.nn.ReLU()
        self.gateA = torch.nn.Linear(8, 1)
        self.gateA_activation = torch.nn.Sigmoid() # Sigmoid for the gating mechanism

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Defines the forward pass of the AmplitudePolaritiesModel.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input tensors:
                                             - "XYZ": Event coordinates (batch_size, 3).
                                             - "Amplitudes": Amplitude measurements (batch_size, num_stations).
                                             - "Polarities": Polarity measurements (batch_size, num_stations).
                                             - "staz_pos": Station positions (batch_size, num_stations, 2).

        Returns:
            torch.Tensor: A tensor containing the predicted strike, dip, and rake angles
                          represented as sine and cosine components (batch_size, 6).
                          (2 for strike, 2 for dip, 2 for rake).
        """
        xyz = batch["XYZ"]
        Amplitudes = batch["Amplitudes"]
        polarities = batch["Polarities"].unsqueeze(1) # Add a channel dimension for polarities

        # Normalize amplitudes based on their maximum value in each sample.
        Amplitudes_maxs = Amplitudes.max(1).values.unsqueeze(1)
        # Handle cases where amplitude is -1 (missing data) by keeping them as -1.
        Amplitudes_normalized = torch.where(
            Amplitudes == -1, -1, Amplitudes / Amplitudes_maxs
        ).unsqueeze(1)

        # Create a presence vector indicating where amplitudes or polarities are valid.
        presence_vector = (batch["Amplitudes"] != -1).float().unsqueeze(1)
        presence_vector = presence_vector + (
            batch["Polarities"] != 0
        ).float().unsqueeze(1)
        
        # Scale station positions using the pre-initialized MinMaxScalerLayer for XY.
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1) # Permute to (batch_size, 2, num_stazioni) for concatenation

        # Process XYZ coordinates through a series of fully connected layers.
        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1) # Final XYZ embedding

        # Combine amplitude-related features for the amplitude branch.
        # Concatenate normalized amplitudes, presence vector, scaled station positions, and XYZ embedding.
        Amplitudes_combined = torch.cat(
            [Amplitudes_normalized, presence_vector, staz_pos, x], dim=1
        )  # Resulting shape: (batch_size, 5, num_stazioni)

        # Combine polarity-related features for the polarity branch.
        # Concatenate polarities, presence vector, scaled station positions, and XYZ embedding.
        Polarities_combined = torch.cat(
            [polarities, presence_vector, staz_pos, x], dim=1
        ) # Resulting shape: (batch_size, 5, num_stazioni)

        # Permute dimensions for Multihead Attention: (batch_size, num_stazioni, features)
        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        
        # Apply Multihead Attention to Amplitude features.
        Amplitudes_combined, _ = self.multihead_attention1(
            Amplitudes_combined, Amplitudes_combined, Amplitudes_combined
        )
        # Apply Multihead Attention to Polarity features.
        Polarities_combined, _ = self.multihead_attention2(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        # Permute back dimensions to (batch_size, features, num_stazioni) for convolutions.
        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        # Apply dilated 1D convolutions to Amplitude and Polarity features.
        A_outputs = [conv(Amplitudes_combined) for conv in self.Aconvs]
        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        # Flatten the output of each convolution.
        A_outputs = [self.flattenA(co) for co in A_outputs]
        P_outputs = [self.flattenP(co) for co in P_outputs]

        # Concatenate the flattened outputs from different dilation convolutions.
        yA = torch.cat(A_outputs, dim=1) # Combined Amplitude features
        yP = torch.cat(P_outputs, dim=1) # Combined Polarity features

        # Process Polarity features through a series of fully connected layers.
        yP = self.relu1cp0(self.fc1cp0(yP))
        yP = self.relu1cp(self.fc1cp(yP))
        yP = self.relu2cp(self.fc2cp(yP))
        yP = self.relu3cp(self.fc3cp(yP))
        yP = self.relu4cp(self.fc4cp(yP))
        yP = self.relu5cp(self.fc5cp(yP))
        yP = self.relu6cp(self.fc6cp(yP))
        yP = self.relu7cp(self.fc7cp(yP))

        # Process Amplitude features through a series of fully connected layers for gating.
        yA = self.relu00A(self.fc00A(yA))
        yA = self.relu0A(self.fc0A(yA))
        yA = self.relu1A(self.fc1A(yA))
        yA = self.relu2A(self.fc2A(yA))
        # Compute the gating value using a sigmoid activation.
        gate = self.gateA_activation(self.gateA(yA))
        
        # Combine amplitude and polarity features using a gating mechanism.
        # The output 'x' is a weighted sum of yA (Amplitude) and yP (Polarity).
        x = yA * gate + yP 

        # Final fully connected layer for the combined features.
        x = self.relu8cp(self.fc8cp(x))

        # Predict strike angle components (sin and cos).
        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        
        # Predict dip angle components.
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        
        # Predict rake angle components (sin and cos, with different activations).
        rake = self.fc_rake(x)
        rake_sin = self.fc_rake_activation1(rake[..., :1]) # First element for sin(rake)
        rake_cos = self.fc_rake_activation2(rake[..., 1:]) # Second element for cos(rake)
        
        # Concatenate all predicted components: [strike_sin, strike_cos, dip_sin, dip_cos, rake_sin, rake_cos]
        x = torch.cat([strike, dip, rake_sin, rake_cos], dim=1)

        return x

    def save_parameters_correctly(self, path: str, verbose: bool = True):
        """
        Saves the model's state dictionary to the specified path.
        Handles the PyTorch Generator object to avoid serialization issues.

        Args:
            path (str): The file path to save the model parameters.
            verbose (bool, optional): If True, prints a message upon successful saving. Defaults to True.
        """
        # Temporarily set generator to None to prevent issues during deepcopy and serialization.
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        # Create a deep copy of the model for saving, ensuring no references to the original generator and preserving the model for the training process.
        model_copy = deepcopy(self)

        # Restore the generator on the original model.
        if hasattr(self, "generator"):
            self.generator = generator_backup

        # Move the copied model to CPU before saving to ensure compatibility across devices.
        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        # Delete the copied model to free up memory.
        del model_copy

    def load_parameters_correctly(self, path: str, device: Union[torch.device, None] = None, verbose: bool = True):
        """
        Loads the model's state dictionary from the specified path.

        Args:
            path (str): The file path from which to load the model parameters.
            device (Union[torch.device, None], optional): The device to load the model onto.
                                                         If None, uses the model's current device. Defaults to None.
            verbose (bool, optional): If True, prints a message upon successful loading. Defaults to True.
        """
        # Determine the target device.
        if device is None:
            device = self.device
        
        # Move the model to CPU before loading the state dictionary to prevent potential device mismatches.
        self.cpu()
        self.load_state_dict(torch.load(path))
        # Move the model to the specified (or current) device after loading.
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(
        self,
        batch: Dict[str, torch.Tensor],
        zeroing_value: Union[float, List[float]],
        switch_sign: float = 0,
        normal_noise: float = 0,
        beta_total_concentration: float = 0.,
        inplace: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs various preprocessing steps on the input batch, including
        zeroing out values, switching polarity signs, and adding normal noise.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing "Amplitudes" and "Polarities" tensors.
            zeroing_value (Union[float, List[float]]): Controls the amount of zeroing.
                                                        If float, same percentage of polarities and amplitudes to zero.
                                                        If list [float, float], percentage for polarities and amplitudes.
            switch_sign (float, optional): Percentage of polarities to switch sign. Defaults to 0.
            normal_noise (float, optional): Standard deviation multiplier for normal noise added to amplitudes. Defaults to 0.
            beta_total_concentration (float, optional): Parameter for beta distribution if zeroing_value is a list. Defaults to 0.0.
            inplace (bool, optional): If True, modifies the batch in place. Otherwise, returns a new batch. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: The preprocessed batch dictionary.
        """
        # Validate input batch structure.
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Amplitudes" in batch, "The batch must contain the Amplitudes field"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        # Clone amplitude and polarity tensors to avoid modifying original batch if inplace is False.
        if not isinstance(batch["Amplitudes"], torch.Tensor) or not isinstance(
            batch["Polarities"], torch.Tensor
        ):
            Xa = torch.tensor(batch["Amplitudes"]).clone()
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xa = batch["Amplitudes"].clone()
            Xp = batch["Polarities"].clone()

        # Validate tensor shapes and amplitude positivity.
        assert (
            Xa.shape == Xp.shape
        ), "Amplitudes and Polarities must have the same shape"
        assert torch.all(Xa == Xa.abs()).item(), "Amplitudes must be positive"

        device = Xa.device
        # Initialize generator if not already set.
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        # Handle float zeroing_value (for polarities only).
        if isinstance(zeroing_value, float) and zeroing_value > 0:
            assert zeroing_value < 1, "The zeroing_value must be in the range [0, 1)"

            # Apply zeroing to polarities.
            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            # Set corresponding amplitudes to -1 and polarities to 0.
            Xa[rows, cols] = -1
            Xp[rows, cols] = 0

        # Handle list zeroing_value (for both polarities and amplitudes).
        elif isinstance(zeroing_value, list):
            assert (
                len(zeroing_value) == 2
            ), "The zeroing_value must be a list of two elements"

            # Zeroing where polarities are more heavily masked than amplitudes.
            if zeroing_value[0] > 0 and zeroing_value[1] >= zeroing_value[0]:
                assert (
                    zeroing_value[0] < 1
                ), "The zeroing_value must be in the range [0, 1)"
                assert (
                    zeroing_value[1] < 1
                ), "The zeroing_value must be in the range [0, 1)"

                # Mark amplitudes as -1 where polarities are 0.
                Xa = torch.where(Xp == 0, -1, Xa)
                # Apply joint zeroing.
                rowsP, colsP, rowsA, colsA = self.zeroing_as_same(
                    zeroing_value, Xp, Xa, beta_total_concentration, self.generator
                )

                # Apply zeroing to polarities and amplitudes.
                Xp[rowsP, colsP] = 0
                Xa[rowsA, colsA] = -1

            # Zeroing for polarities only (first element of list).
            elif zeroing_value[0] > 0:
                assert (
                    zeroing_value[0] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(zeroing_value[0], Xp, self.generator)

                Xp[rows, cols] = 0

            # Zeroing for amplitudes only (second element of list).
            elif zeroing_value[1] > 0:
                assert (
                    zeroing_value[1] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(
                    zeroing_value[1], Xa, self.generator, none_value=-1
                )

                Xa[rows, cols] = -1

        # No zeroing applied.
        elif zeroing_value == 0 or zeroing_value is None:
            pass

        else:
            raise ValueError(
                "The zeroing_value must be a float or a list of two floats"
            )

        # Apply sign switching to polarities.
        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        # Add normal noise to amplitudes.
        if normal_noise > 0:
            Xa += torch.normal(
                torch.tensor(0.0, device=self.device),
                normal_noise * torch.where(Xa == -1, 0, Xa), # Noise only applied to non-missing amplitudes
                generator=self.generator,
            )

        # Update the batch either in-place or return a new copy.
        if inplace:
            batch["Amplitudes"] = Xa
            batch["Polarities"] = Xp
            return batch
        else:
            batch_copy = deepcopy(batch)
            batch_copy["Amplitudes"] = Xa
            batch_copy["Polarities"] = Xp
            return batch_copy

    @staticmethod
    def predict_angles(outputs: torch.Tensor, data_format: str = "sdr") -> torch.Tensor:
        """
        Converts trigonometric outputs (sine and cosine components) back into angles in degrees.

        Args:
            outputs (torch.Tensor): A tensor containing the sine and cosine components of the angles.
                                    Expected shape depends on `data_format`.
            data_format (str, optional): The format of the input `outputs`.
                                         "sdr" for [s1, c1, s2, c2, s3, c3] (strike, dip, rake).
                                         "sin_cos" for [s1, s2, s3, c1, c2, c3].
                                         Defaults to "sdr".

        Returns:
            torch.Tensor: A tensor of predicted angles in degrees, shaped [batch_size, 3].
                          (Strike, Dip, Rake).
        """
        if data_format == "sdr":
            # Extract sine components (every second element starting from the first)
            sin_theta = outputs[:, ::2]
            # Extract cosine components (every second element starting from the second)
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            # Extract sine components (first half of the features)
            sin_theta = outputs[:, :3]
            # Extract cosine components (second half of the features)
            cos_theta = outputs[:, 3:]
        
        # Calculate angles in radians using atan2, which handles all quadrants.
        angles_rad = torch.atan2(sin_theta, cos_theta)
        # Convert angles from radians to degrees.
        angles_deg = torch.rad2deg(angles_rad)  
        # Ensure strike angle (first column) is within [0, 360) degrees.
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets: torch.Tensor, data_format: str = "sdr") -> torch.Tensor:
        """
        Converts target angles from degrees to their sine and cosine components.

        Args:
            targets (torch.Tensor): A tensor of target angles in degrees, shaped [batch_size, 3]
                                    (Strike, Dip, Rake).
            data_format (str, optional): The desired output format of the trigonometric components.
                                         "sdr" for [s1, c1, s2, c2, s3, c3].
                                         "sin_cos" for [s1, s2, s3, c1, c2, c3].
                                         Defaults to "sdr".

        Returns:
            torch.Tensor: A tensor containing the sine and cosine components of the target angles,
                          shaped [batch_size, 6].
        """
        # Convert target angles from degrees to radians.
        targets_rad = torch.deg2rad(targets)  
        # Calculate sine components of the angles.
        sin_targets = torch.sin(targets_rad)
        # Calculate cosine components of the angles.
        cos_targets = torch.cos(targets_rad)
        
        if data_format == "sdr":
            # Stack sine and cosine components for each angle and flatten to [s1, c1, s2, c2, s3, c3].
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            # Concatenate all sine components, then all cosine components to [s1, s2, s3, c1, c2, c3].
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # [batch_size, 6]

    @staticmethod
    def zeroing(
        zeroing_value: float, 
        X: torch.Tensor, 
        generator: Union[torch.Generator, None] = None, 
        device: Union[torch.device, None] = None, 
        none_value: Union[float, int] = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly zeroes out a percentage of non-`none_value` elements in a tensor.

        Args:
            zeroing_value (float): The percentage (0 to 1) of non-`none_value` elements to zero out.
            X (torch.Tensor): The input tensor.
            generator (Union[torch.Generator, None], optional): A PyTorch random number generator. Defaults to None.
            device (Union[torch.device, None], optional): The device to perform operations on. Defaults to None.
            none_value (Union[float, int], optional): The value that represents 'no data' or 'zero'.
                                                      Elements equal to this value are ignored. Defaults to 0.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                               - rows: The row indices of the elements to be zeroed.
                                               - cols: The column indices of the elements to be zeroed.
        """
        # Ensure tensor is on the correct device.
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        # Initialize generator if not provided.
        if generator is None:
            generator = torch.Generator(device=device)

        # Calculate the number of non-`none_value` elements per row.
        Z = (X != none_value).sum(dim=1)
        # Generate random numbers for each row to determine how many elements to zero.
        randomizer = torch.rand(Z.shape, device=device)
        # Calculate the number of elements to sample for zeroing in each row.
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        # Get the maximum number of elements to sample across all rows.
        Z_sampler_max = Z_sampler.max().item()
        
        # Create a probability distribution for sampling: 1 for non-`none_value` elements, 0 otherwise.
        P = torch.where(X != none_value, 1, 0).float()
        # Normalize probabilities per row.
        P /= P.sum(dim=1, keepdim=True)
        # Sample indices of elements to zero out. `Z_sampler_max` ensures enough indices are sampled for all rows.
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        # Create row indices for the elements to be zeroed.
        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        # Create a mask to select the correct number of sampled indices for each row.
        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        # Apply the mask to get the column indices of the elements to be zeroed.
        cols = idxs[mask]
        return rows, cols

    @staticmethod
    def zeroing_as_same(
        zeroing_values: List[float], 
        Xp: torch.Tensor, 
        Xa: torch.Tensor, 
        beta_total_concentration: float = 0.0, 
        generator: Union[torch.Generator, None] = None, 
        device: Union[torch.device, None] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly zeroes out a percentage of elements in both polarity (Xp) and amplitude (Xa)
        tensors, ensuring a consistent masking pattern across both.

        Args:
            zeroing_values (List[float]): A list of two floats [polarity_zeroing_percentage, amplitude_zeroing_percentage].
            Xp (torch.Tensor): The input tensor for polarities.
            Xa (torch.Tensor): The input tensor for amplitudes.
            beta_total_concentration (float, optional): Parameter for beta distribution if > 2. Defaults to 0.0.
            generator (Union[torch.Generator, None], optional): A PyTorch random number generator. Defaults to None.
            device (Union[torch.device, None], optional): The device to perform operations on. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                                                                           - rowsP: Row indices for polarities to be zeroed.
                                                                           - colsP: Column indices for polarities to be zeroed.
                                                                           - rowsA: Row indices for amplitudes to be zeroed.
                                                                           - colsA: Column indices for amplitudes to be zeroed.
        """
        assert (
            len(zeroing_values) == 2
        ), "The zeroing_values must be a list of two elements"
        
        # Ensure tensors are on the same device and move if necessary.
        if device is None:
            assert Xp.device == Xa.device, "Xp and Xa must be on the same device"
            device = Xa.device
        elif device != Xa.device or device != Xp.device:
            Xa = Xa.to(device)
            Xp = Xp.to(device)

        # Initialize generator if not provided.
        if generator is None:
            generator = torch.Generator(device=device)

        # Calculate the number of non-zero polarities and non-negative amplitudes per row.
        Zp = (Xp != 0).sum(dim=1)
        Za = (Xa != -1).sum(dim=1) # Assuming -1 indicates no amplitude data.
        
        # Generate randomizers using a Beta distribution if beta_total_concentration is specified,
        # otherwise use uniform random numbers.
        if beta_total_concentration > 2:
            dist = _beta_distribution(beta_total_concentration, 1)
            randomizerP = dist.sample(Zp.shape).to(device) * zeroing_values[0]
            randomizerA = dist.sample(Za.shape).to(device) * zeroing_values[1]
        else:
            randomizerP = torch.rand(Zp.shape, device=device) * zeroing_values[0]
            randomizerA = torch.rand(Za.shape, device=device) * zeroing_values[1]

        # Calculate the number of elements to sample for zeroing for polarities and amplitudes.
        Z_samplerP = torch.ceil(Zp * randomizerP).to(torch.int32)
        Z_samplerA = torch.ceil(Za * randomizerA).to(torch.int32)
        
        # Ensure Z_samplerA is at least Z_samplerP to zero out at least as many amplitudes as polarities.
        Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        # Get the maximum number of elements to sample for amplitudes across all rows.
        Z_sampler_max = Z_samplerA.max().item()
        # Assertion to ensure a valid sampling range.
        assert (
            Z_sampler_max < Za.max().item()
        ), "Z_sampler_max must be less than Za.max().item()"

        # Create a probability distribution for sampling indices based on non-zero polarities.
        probs = torch.where(Xp != 0, 1, 0).float()
        # Normalize probabilities per row.
        probs /= probs.sum(dim=1, keepdim=True)
        # Sample indices for zeroing based on polarity presence.
        idxs = torch.multinomial(probs, Z_sampler_max, generator=generator)

        # Create row indices for polarities to be zeroed.
        rowsP = torch.arange(Z_samplerP.size(0), device=device).repeat_interleave(
            Z_samplerP
        )
        # Create row indices for amplitudes to be zeroed.
        rowsA = torch.arange(Z_samplerA.size(0), device=device).repeat_interleave(
            Z_samplerA
        )

        # Create masks to select the correct number of sampled indices for each row for polarities and amplitudes.
        maskP = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerP.unsqueeze(1)
        maskA = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerA.unsqueeze(1)

        # Apply masks to get column indices for polarities and amplitudes.
        colsP = idxs[maskP]
        colsA = idxs[maskA]
        return rowsP, colsP, rowsA, colsA


# %%

class OnlyPolarityModel(pl.LightningModule):
    def __init__(
        self,
        n_stations,
        xyz_boundary,
        scaling_range=None,
        generator=None,
    ):
        super().__init__()
        self.n_stations = n_stations        
        self.xyz_boundary = xyz_boundary

        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.relu10xyz = torch.nn.ReLU()

        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        self.flattenP = nn.Flatten()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )
        self.fc1p = torch.nn.LazyLinear(1024)
        self.relu1p = torch.nn.ReLU()
        self.fc2p = torch.nn.Linear(1024, 512)
        self.relu2p = torch.nn.ReLU()
        self.fc3p = torch.nn.Linear(512, 256)
        self.relu3p = torch.nn.ReLU()
        self.fc4p = torch.nn.Linear(256, 128)
        self.relu4p = torch.nn.ReLU()
        self.fc5p = torch.nn.Linear(128, 64)
        self.relu5p = torch.nn.ReLU()
        self.fc6p = torch.nn.Linear(64, 32)
        self.relu6p = torch.nn.ReLU()
        self.fc7p = torch.nn.Linear(32, 16)
        self.relu7p = torch.nn.ReLU()
        self.fc8p = torch.nn.Linear(16, 8)
        self.relu8p = torch.nn.ReLU()
        self.fc9p = torch.nn.Linear(8, self.output_shape)
        self.relu9p = torch.nn.Tanh()

        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0)
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation = torch.nn.Tanh()

    def forward(self, batch):
        xyz = batch["XYZ"]
        polarities = batch["Polarities"].unsqueeze(1)
        presence_vector = (batch["Polarities"] != 0).float().unsqueeze(1)
        staz_pos = batch["staz_pos"]
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1)

        Polarities_combined = torch.cat(
            [polarities, presence_vector, staz_pos, x], dim=1
        )

        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        Polarities_combined, _ = self.multihead_attention(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        P_outputs = [self.flattenP(co) for co in P_outputs]

        yP = torch.cat(P_outputs, dim=1)

        yP = self.relu1p(self.fc1p(yP))
        yP = self.relu2p(self.fc2p(yP))
        yP = self.relu3p(self.fc3p(yP))
        yP = self.relu4p(self.fc4p(yP))
        yP = self.relu5p(self.fc5p(yP))
        yP = self.relu6p(self.fc6p(yP))
        yP = self.relu7p(self.fc7p(yP))
        yP = self.relu8p(self.fc8p(yP))

        x = yP

        x = self.relu9p(self.fc9p(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake = self.fc_rake_activation(rake)
        x = torch.cat([strike, dip, rake], dim=1)

        return x

    def save_parameters_correctly(self, path, verbose=True):
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        model_copy = deepcopy(self)

        if hasattr(self, "generator"):
            self.generator = generator_backup

        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        del model_copy

    def load_parameters_correctly(self, path, device=None, verbose=True):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(self, batch, zeroing_value, switch_sign=0, *args, **kwargs):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        if not isinstance(batch["Polarities"], torch.Tensor):
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xp = batch["Polarities"].clone()

        device = Xp.device
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        assert isinstance(zeroing_value, float), "The zeroing_value must be a float"
        if zeroing_value > 0:
            assert zeroing_value <= 1, "The zeroing_value must be in the range [0, 1]"

            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            Xp[rows, cols] = 0

        else:
            pass

        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        if isinstance(batch, dict):
            batch["Polarities"] = Xp
            return batch
        else:
            return Xp

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        targets_rad = torch.deg2rad(targets) 
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  

    @staticmethod
    def zeroing(zeroing_value, X, generator=None, device=None):
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Z = (X != 0).sum(dim=1)
        randomizer = torch.rand(Z.shape, device=device)
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        Z_sampler_max = Z_sampler.max().item()
        P = torch.where(X != 0, 1, 0).float()
        P /= P.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        cols = idxs[mask]
        return rows, cols

# %%
