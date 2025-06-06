#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import torch
from shapely.geometry import Polygon
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator

from .dictattr import dictattr

from .synthetic import (
    ensure_float_dtype,
    read_velmod_data,
    read_stations_data,
    gclc,
    azimuth,
    takeoff,
    radpar,
    compute_probs,
)


def extract_grid(config: dictattr, device: torch.device) -> torch.Tensor:
    """
    Generate a grid of values based on configuration parameters, excluding specific values.

    The grid is constructed from 'config.min' to 'config.max' with increments of 'config.step'.
    A small fraction (10% of the step) is added to 'config.max' to ensure inclusion of the maximum
    value if it falls on the grid. Then, any values specified in 'config.exclude' are removed from the grid.

    Args:
        config (dictattr): Configuration object with attributes:
            - min (int or float): The starting value of the grid.
            - max (int or float): The ending value of the grid.
            - step (int or float): The increment between consecutive grid values.
            - exclude (list): A list of values to be excluded from the grid.
        device (torch.device): The device on which the grid will be created.

    Returns:
        Tensor: A 1D tensor containing the grid values, with excluded values removed.
    """
    # Create a grid from config.min to config.max (inclusive) using a step size.
    grid = torch.arange(
        config.min,
        config.max
        + config.step * 0.1,  # Add a fraction of the step to include config.max.
        config.step,
        device=device,
    )
    # Remove values that are specified in config.exclude.
    for q in config.exclude:
        # Create a tensor for the value 'q' on the given device.
        grid = grid[~torch.isin(grid, torch.tensor(q, device=device))]
    return grid


def get_random_uniform(
    grid: torch.Tensor, size: int, generator: torch.Generator
) -> torch.Tensor:
    """
    Randomly sample elements uniformly from a provided grid.

    The function uses multinomial sampling with replacement over a uniform weight vector
    (i.e., each element has an equal probability of being chosen).

    Args:
        grid (Tensor): A 1D tensor of values from which to sample.
        size (int): The number of samples to draw.
        generator (torch.Generator): A random number generator for reproducibility.

    Returns:
        Tensor: A tensor containing 'size' randomly selected values from the grid.
    """
    # Create a uniform weight vector for the grid.
    weights = torch.ones(len(grid), device=grid.device)
    # Sample indices from the grid using multinomial sampling with replacement.
    indices = torch.multinomial(weights, size, generator=generator, replacement=True)
    # Retrieve the grid values corresponding to the sampled indices.
    return grid[indices.to(grid.device)]


class FocalOnlineDataset:
    """
    A dataset class for generating synthetic focal mechanism data on the fly.

    This class generates synthetic data based on specified boundaries and steps for both
    spatial (x, y, z) and focal mechanism (strike, dip, rake) parameters. It also reads
    velocity model and station data, computes station coordinates in the local (NED) system,
    and constructs grids for sampling the parameters.
    """

    def __init__(
        self,
        xyz_boundary: list[float],
        sdr_boundary: list[float],
        xyz_steps: list[float],
        sdr_steps: list[float],
        exclude_strike: list[float],
        exclude_dip: list[float],
        exclude_rake: list[float],
        velmod_file: str,
        stations_file: str,
        input_dir: Path,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        xyz_translation_step_percs: list[float] | None = None,
        xyz_scaling_range: list[tuple[float, float]] | None = None,
        float_type: str = "float32",
        polygon: Polygon | torch.Tensor | None = None,
        xyz_points: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the FocalOnlineDataset.

        Args:
            xyz_boundary (list[float]): [x_min, x_max, y_min, y_max, z_min, z_max].
            sdr_boundary (list[float]): [strike_min, strike_max, dip_min, dip_max, rake_min, rake_max].
            xyz_steps (list[float]): Step sizes for x, y, and z.
            sdr_steps (list[float]): Step sizes for strike, dip, and rake.
            exclude_strike (list[float]): Values to exclude from the strike grid.
            exclude_dip (list[float]): Values to exclude from the dip grid.
            exclude_rake (list[float]): Values to exclude from the rake grid.
            velmod_file (str): Filename for the velocity model.
            stations_file (str): Filename for the stations data.
            input_dir (Path): Directory containing the data files.
            batch_size (int): Size of the batch to generate.
            generator (torch.Generator): Random number generator for reproducibility.
            device (torch.device): The torch device to perform computations on.
            xyz_translation_step_percs (list[float] | None): Optional translation step percentages for x, y, and z.
            xyz_scaling_range (list[tuple[float, float]] | None): Optional scaling ranges for x, y, and z.
            float_type (str): Floating point precision (e.g., "float32").
            polygon (Polygon | torch.Tensor | None): Optional polygon for spatial constraints.
            xyz_points (torch.Tensor | None): Optional tensor of xyz points. When provided, generates uses the coordinates provided that are within the polygon and the imposed boundary.
        """
        # Save the initialization parameters as attributes
        self.input_dir = input_dir
        self.xyz_boundary = xyz_boundary
        self.sdr_boundary = sdr_boundary
        self.xyz_steps = xyz_steps
        self.sdr_steps = sdr_steps
        self.exclude_strike = exclude_strike
        self.exclude_dip = exclude_dip
        self.exclude_rake = exclude_rake
        self.velmod_file = velmod_file
        self.stations_file = stations_file
        self.batch_size = batch_size
        self.xyz_scaling_range = xyz_scaling_range
        self.generator = generator
        self.device = device
        self.float_type = float_type
        self.xyz_translation_step_percs = xyz_translation_step_percs
        self.polygon = polygon
        self.xyz_points = xyz_points

        # Remove local variables to avoid accidental use later.
        del (
            input_dir,
            xyz_boundary,
            sdr_boundary,
            xyz_steps,
            sdr_steps,
            exclude_strike,
            exclude_dip,
            exclude_rake,
            velmod_file,
            stations_file,
            xyz_translation_step_percs,
            batch_size,
            generator,
            xyz_scaling_range,
            device,
            float_type,
            polygon,
        )

        if self.xyz_points is not None:
            assert isinstance(
                self.xyz_points, torch.Tensor
            ), "xyz_points must be a tensor"
            assert (
                self.xyz_points.shape[1] == 3
            ), "xyz_points must be of shape [n_points, 3]"
            assert self.xyz_points.shape[0] > 0, "xyz_points must have at least 1 point"
            if len(self.xyz_points) < self.batch_size:
                print("Not enough points in polygon, using all points with repetition")

        # ----------------- Validation Checks -----------------
        # Validate translation step percentages if provided.
        if self.xyz_translation_step_percs is not None:
            if len(self.xyz_translation_step_percs) != 3:
                raise ValueError("Translation steps must have length 3")
            if any(
                not isinstance(translation_step_perc, (int, float))
                or translation_step_perc < 0
                or translation_step_perc >= 1
                for translation_step_perc in self.xyz_translation_step_percs
            ):
                raise ValueError(
                    "Translation steps must be between 0 and 1, 1 excluded"
                )

        # Validate scaling range if provided.
        if self.xyz_scaling_range is not None:
            if len(self.xyz_scaling_range) != 3:
                raise ValueError("Scaling ranges must have length 3")
            if any(len(scaling_range) != 2 for scaling_range in self.xyz_scaling_range):
                raise ValueError(
                    "Each element of xyz scaling range must be of length 2"
                )
            if any(
                not isinstance(scaling_min, (int, float))
                or not isinstance(scaling_max, (int, float))
                or scaling_min >= scaling_max
                for scaling_min, scaling_max in self.xyz_scaling_range
            ):
                raise ValueError("xyz scaling min must be smaller than scaling max")

        # Ensure the batch size is a positive integer.
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        # ----------------- File Paths and Data Types -----------------
        # Construct full file paths for velocity model and station data.
        self.velmod_path = self.input_dir / self.velmod_file
        self.stations_path = self.input_dir / self.stations_file

        # Define numpy and torch data types based on the provided float type.
        self.np_dtype = np.dtype(self.float_type)
        self.torch_dtype = ensure_float_dtype(self.float_type)

        # ----------------- Load Velocity Model Data -----------------
        # Read the velocity model data from file.
        self.latref, self.lonref, self.v, self.d = read_velmod_data(
            self.velmod_path, self.torch_dtype
        )

        # Move velocity model data to the specified device.
        self.v = self.v.to(self.device)
        self.d = self.d.to(self.device)

        # ----------------- Load Station Data -----------------
        # Read station information from file.
        self.stations, self.lats, self.lons, _ = read_stations_data(
            self.stations_path, self.torch_dtype
        )

        # Compute station coordinates (in local NED system) using provided geolocation conversion.
        self.ystaz, self.xstaz = gclc(self.latref, self.lonref, self.lats, self.lons)
        self.xstaz = self.xstaz.to(self.device)
        self.ystaz = self.ystaz.to(self.device)

        self.xyz_points = (
            self.xyz_points.to(self.device) if self.xyz_points is not None else None
        )

        # ----------------- Build Grid Configuration -----------------
        # --- Handle optional polygon input and adjust spatial boundaries accordingly ---
        if isinstance(self.polygon, Polygon):
            # Convert a Shapely Polygon into a (n_vertices, 2) torch tensor of [x, y] coordinates
            self.polygon = torch.tensor(
                np.array(self.polygon.exterior.coords.xy).T, dtype=self.torch_dtype
            ).to(self.device)
        elif isinstance(self.polygon, torch.Tensor):
            # Validate that tensor is of shape [n_points, 2]
            if self.polygon.shape[1] != 2:
                raise ValueError("Polygon must be of shape [n_points, 2]")
            # Ensure there are at least three points to form a polygon
            if self.polygon.shape[0] < 3:
                raise ValueError("Polygon must have at least 3 points")
            # Move the tensor to the target device
            self.polygon = self.polygon.to(self.device)

        if self.polygon is not None and self.xyz_points is None:
            # Adjust x_min if it's outside the polygon's minimum x
            if self.xyz_boundary[0] < self.polygon[:, 0].min():
                self.xyz_boundary[0] = self.polygon[:, 0].min()
                print(
                    f"xyz_boundary[0] (x_min) set to {self.polygon[:, 0].min()} because of polygon"
                )
            # Adjust x_max if it's outside the polygon's maximum x
            if self.xyz_boundary[1] > self.polygon[:, 0].max():
                self.xyz_boundary[1] = self.polygon[:, 0].max()
                print(
                    f"xyz_boundary[1] (x_max) set to {self.polygon[:, 0].max()} because of polygon"
                )
            # Adjust y_min based on polygon's minimum y
            if self.xyz_boundary[2] < self.polygon[:, 1].min():
                self.xyz_boundary[2] = self.polygon[:, 1].min()
                print(
                    f"xyz_boundary[2] (y_min) set to {self.polygon[:, 1].min()} because of polygon"
                )
            # Adjust y_max based on polygon's maximum y
            if self.xyz_boundary[3] > self.polygon[:, 1].max():
                self.xyz_boundary[3] = self.polygon[:, 1].max()
                print(
                    f"xyz_boundary[3] (y_max) set to {self.polygon[:, 1].max()} because of polygon"
                )
        elif self.polygon is not None and self.xyz_points is not None:
            assert isinstance(
                self.xyz_points, torch.Tensor
            ), "xyz_points must be a tensor"
            assert (
                self.xyz_points.shape[1] == 3
            ), "xyz_points must be of shape [n_points, 3]"
            assert self.xyz_points.shape[0] > 0, "xyz_points must have at least 1 point"

            points = self.xyz_points[:, :2]
            mask = self.point_in_polygon(points, self.polygon)
            if len(mask) != len(points):
                print(f"Find {len(points) - len(mask)} points not in polygon")
                self.xyz_points = self.xyz_points[mask]
                assert len(self.xyz_points) > 0, "No points in polygon"
                if len(self.xyz_points) < self.batch_size:
                    print(
                        "Not enough points in polygon after selection, using all points with repetition"
                    )

        # Create a configuration dictionary for the grids using provided boundaries and steps.
        config = dict(
            x=dict(
                min=self.xyz_boundary[0],
                max=self.xyz_boundary[1],
                step=self.xyz_steps[0],
                exclude=[],
            ),
            y=dict(
                min=self.xyz_boundary[2],
                max=self.xyz_boundary[3],
                step=self.xyz_steps[1],
                exclude=[],
            ),
            z=dict(
                min=self.xyz_boundary[4],
                max=self.xyz_boundary[5],
                step=self.xyz_steps[2],
                exclude=[],
            ),
            strike=dict(
                min=self.sdr_boundary[0],
                max=self.sdr_boundary[1],
                step=self.sdr_steps[0],
                exclude=self.exclude_strike,
            ),
            dip=dict(
                min=self.sdr_boundary[2],
                max=self.sdr_boundary[3],
                step=self.sdr_steps[1],
                exclude=self.exclude_dip,
            ),
            rake=dict(
                min=self.sdr_boundary[4],
                max=self.sdr_boundary[5],
                step=self.sdr_steps[2],
                exclude=self.exclude_rake,
            ),
        )

        # Convert the plain dictionary to a dictattr for dot-access.
        config = dictattr(config)

        # ----------------- Compute Translation Adjustments -----------------
        # If translation percentages are provided, use them; otherwise, default to (0,0,0)
        (transl_step_perc_x, transl_step_perc_y, transl_step_perc_z) = (
            (0.0, 0.0, 0.0)
            if self.xyz_translation_step_percs is None
            else self.xyz_translation_step_percs
        )

        # Adjust the spatial grid boundaries based on the translation step percentages.
        x_min = config.x.min + transl_step_perc_x * config.x.step
        x_max = config.x.max + transl_step_perc_x * config.x.step

        y_min = config.y.min + transl_step_perc_y * config.y.step
        y_max = config.y.max + transl_step_perc_y * config.y.step

        z_min = config.z.min + transl_step_perc_z * config.z.step
        z_max = config.z.max + transl_step_perc_z * config.z.step

        # Store the adjusted spatial boundaries as tensors on the specified device.
        self.xyz_min = torch.tensor([x_min, y_min, z_min], dtype=self.torch_dtype).to(
            self.device
        )
        self.xyz_max = torch.tensor([x_max, y_max, z_max], dtype=self.torch_dtype).to(
            self.device
        )

        # ----------------- Compute Scaling Ranges (Optional) -----------------
        if self.xyz_scaling_range is not None:
            (
                (x_scaled_min, x_scaled_max),
                (y_scaled_min, y_scaled_max),
                (z_scaled_min, z_scaled_max),
            ) = self.xyz_scaling_range

            self.xyz_scaled_min = torch.tensor(
                [x_scaled_min, y_scaled_min, z_scaled_min], dtype=self.torch_dtype
            ).to(self.device)
            self.xyz_scaled_max = torch.tensor(
                [x_scaled_max, y_scaled_max, z_scaled_max], dtype=self.torch_dtype
            ).to(self.device)

        # ----------------- Build Parameter Grids -----------------
        # Generate grids for x, y, and z and add the translation adjustments.
        self.x_grid = (
            extract_grid(config.x, self.device) + transl_step_perc_x * config.x.step
        )
        self.y_grid = (
            extract_grid(config.y, self.device) + transl_step_perc_y * config.y.step
        )
        self.z_grid = (
            extract_grid(config.z, self.device) + transl_step_perc_z * config.z.step
        )

        # Generate grids for focal mechanism parameters: strike, dip, and rake.
        self.strike_grid = extract_grid(config.strike, self.device)
        self.dip_grid = extract_grid(config.dip, self.device)
        self.rake_grid = extract_grid(config.rake, self.device)

        if self.xyz_points is not None:
            self.gen_batch = self.gen_batch_from_points
        else:
            self.gen_batch = self.gen_batch_from_xyz

    def __len__(self) -> int:
        """
        Return the batch size as the length of the dataset.
        """
        return self.batch_size

    def gen_batch_from_xyz(self) -> Dict[str, torch.Tensor]:
        """
        Generate a random batch of synthetic focal mechanism data.

        This method samples random values for spatial coordinates (x, y, z) and focal mechanism
        parameters (strike, dip, rake) from their respective grids. It then computes additional
        quantities such as azimuth, station-to-source distances, takeoff angles, and amplitude with sign
        (cp) values. Finally, it applies a preprocessing transformation to the spatial coordinates and
        returns a dictionary containing all computed outputs.

        Returns:
            dict[str, torch.Tensor]: A dictionary with keys:
                - 'XYZ': Preprocessed spatial coordinates (scaled) [batch, 3].
                - 'P': Probability values computed from station distances.
                - 'az': Azimuth values.
                - 'ih': Takeoff angles.
                - 'Cp': Amplitude with sign computed via radpar.
                - 'SDR': Focal mechanism parameters (strike, dip, rake) [batch, 3].
                - 'Polarities': Sign of the amplitude.
                - 'Amplitudes': Absolute amplitudes.
        """
        # Randomly sample spatial coordinates from the x, y, and z grids.
        xs = get_random_uniform(self.x_grid, self.batch_size, self.generator)
        ys = get_random_uniform(self.y_grid, self.batch_size, self.generator)
        zs = get_random_uniform(self.z_grid, self.batch_size, self.generator)

        # Randomly sample focal mechanism parameters from their respective grids.
        strikes = get_random_uniform(self.strike_grid, self.batch_size, self.generator)
        dips = get_random_uniform(self.dip_grid, self.batch_size, self.generator)
        rakes = get_random_uniform(self.rake_grid, self.batch_size, self.generator)

        # Compute azimuth values between station coordinates and the sampled source positions.
        azs = azimuth(self.xstaz, self.ystaz, xs, ys)

        # Compute the horizontal distances ("depis") between the sampled source locations and all stations.
        # This uses broadcasting: xs and ys are reshaped to allow pairwise distance computation.
        depis = torch.sqrt(
            (xs[:, None] - self.xstaz[None, :]) ** 2
            + (ys[:, None] - self.ystaz[None, :]) ** 2
        ).to(dtype=self.torch_dtype)

        # Compute probabilities based on the distances.
        probs = compute_probs(depis)

        # Stack x, y, and z coordinates into a single tensor.
        xyzs = torch.stack((xs, ys, zs), dim=1)

        # Stack strike, dip, and rake values into a single tensor.
        strikediprakes = torch.stack((strikes, dips, rakes), dim=1)

        # --- Filter points to those inside the polygon ---
        if self.polygon is not None:
            # Combine separate x and y coordinate tensors into an (n_points, 2) tensor
            points = torch.stack((xs, ys), dim=1)
            # Determine which points lie inside the polygon (boolean mask)
            mask = self.point_in_polygon(points, self.polygon)
            # Apply mask to filter coordinates and associated data arrays
            xyzs = xyzs[mask]  # keep xyz points inside polygon
            zs = zs[mask]  # keep corresponding z-values
            depis = depis[mask]  # keep corresponding depths
            azs = azs[mask]  # keep corresponding azimuths
            probs = probs[mask]  # keep probability values
            strikediprakes = strikediprakes[mask]  # keep strike-dip-rake parameters

        # Compute takeoff angles using the velocity model and the depth information.
        ihs = takeoff(self.v, self.d, depis, zs)

        # Compute the amplitude with sign (cp) and two unused outputs using the radpar function.
        cp, _, _ = radpar(azs, strikediprakes, ihs)

        # Apply a preprocessing transformation to the spatial coordinates.
        xyzs_scaled = self.preprocess_transform_xyz(xyzs)

        # Determine the polarities based on the sign of the amplitude.
        polarities = torch.sign(cp)

        # Return all outputs as a dictionary, ensuring they are detached from the computation graph
        # and cast to the target torch data type.
        return dict(
            XYZ=xyzs_scaled.detach().to(self.torch_dtype),
            P=probs.detach().to(self.torch_dtype),
            az=azs.detach().to(self.torch_dtype),
            ih=ihs.detach().detach().to(self.torch_dtype),
            Cp=cp.detach().to(self.torch_dtype),
            SDR=strikediprakes.detach().to(self.torch_dtype),
            Polarities=polarities.detach().to(self.torch_dtype),
            Amplitudes=cp.abs().detach().to(self.torch_dtype),
        )

    def gen_batch_from_points(self):
        indices = torch.randperm(self.xyz_points.shape[0])[: self.batch_size]
        xyzs = self.xyz_points[indices]
        while len(xyzs) < self.batch_size:
            indices = torch.randperm(self.xyz_points.shape[0])[
                : self.batch_size - len(xyzs)
            ]
            xyzs = torch.cat((xyzs, self.xyz_points[indices]), dim=0)
        xs = xyzs[:, 0]
        ys = xyzs[:, 1]
        zs = xyzs[:, 2]

        strikes = get_random_uniform(self.strike_grid, self.batch_size, self.generator)
        dips = get_random_uniform(self.dip_grid, self.batch_size, self.generator)
        rakes = get_random_uniform(self.rake_grid, self.batch_size, self.generator)

        azs = azimuth(self.xstaz, self.ystaz, xs, ys)
        depis = torch.sqrt(
            (xs[:, None] - self.xstaz[None, :]) ** 2
            + (ys[:, None] - self.ystaz[None, :]) ** 2
        ).to(dtype=self.torch_dtype)
        probs = compute_probs(depis)
        strikediprakes = torch.stack((strikes, dips, rakes), dim=1)
        ihs = takeoff(self.v, self.d, depis, zs)

        cp, _, _ = radpar(azs, strikediprakes, ihs)

        xyzs_scaled = self.preprocess_transform_xyz(xyzs)

        polarities = torch.sign(cp)
        return dict(
            XYZ=xyzs_scaled.detach().to(self.torch_dtype),
            P=probs.detach().to(self.torch_dtype),
            az=azs.detach().to(self.torch_dtype),
            ih=ihs.detach().detach().to(self.torch_dtype),
            Cp=cp.detach().to(self.torch_dtype),
            SDR=strikediprakes.detach().to(self.torch_dtype),
            Polarities=polarities.detach().to(self.torch_dtype),
            Amplitudes=cp.abs().detach().to(self.torch_dtype),
        )

    def gen_data(self) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic data by evaluating all possible parameter combinations.

        This method creates a complete Cartesian product of the spatial grids (x, y, z) and the focal
        mechanism grids (strike, dip, rake) to generate all possible parameter combinations. It then computes:
        - Azimuth values for each combination.
        - Horizontal distances (and derived probabilities) between each source location and stations.
        - Takeoff angles and the amplitude (cp).
        - Preprocessed spatial coordinates.
        - Polarities and absolute amplitudes.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - 'XYZ': Preprocessed spatial coordinates (scaled) [N, 3].
                - 'P': Probability values computed from distances.
                - 'az': Azimuth values.
                - 'ih': Takeoff angles.
                - 'Cp': Amplitudes with sign values.
                - 'SDR': Focal mechanism parameters (strike, dip, rake) [N, 3].
                - 'Polarities': Sign of the amplitudes.
                - 'Amplitudes': Absolute amplitudes.
        """
        # Generate all combinations (Cartesian product) of the parameter grids.
        if self.xyz_points is not None:
            
            idx_xyz = torch.arange(self.xyz_points.shape[0])
            idxS = torch.arange(len(self.strike_grid))
            idxD = torch.arange(len(self.dip_grid))
            idxR = torch.arange(len(self.rake_grid))

            ijk = torch.cartesian_prod(idx_xyz, idxS, idxD, idxR)  # shape (Na*Nb*Nc, 4)

            A = self.xyz_points[ ijk[:, 0] ]             
            b = self.strike_grid[ ijk[:, 1] ].unsqueeze(1) 
            c = self.dip_grid[ ijk[:, 2] ].unsqueeze(1) 
            d = self.rake_grid[ ijk[:, 3] ].unsqueeze(1)

            combinations = torch.cat([A, b, c, d], dim=1)

        else:
            combinations = torch.cartesian_prod(
                self.x_grid,
                self.y_grid,
                self.z_grid,
                self.strike_grid,
                self.dip_grid,
                self.rake_grid,
            )
        if self.polygon is not None:
            raise NotImplementedError("Cannot generate data with polygon")

        # Compute azimuth values for each combination using the station coordinates.
        azs = azimuth(self.xstaz, self.ystaz, combinations[:, 0], combinations[:, 1])

        # Compute horizontal distances (depis) from the source locations (first two columns of combinations)
        # to the station positions using broadcasting.
        depis = torch.sqrt(
            (combinations[:, 0:1] - self.xstaz[None, :]) ** 2
            + (combinations[:, 1:2] - self.ystaz[None, :]) ** 2
        ).to(dtype=self.torch_dtype)

        # Compute probabilities from the distances.
        probs = compute_probs(depis)

        # Stack x, y, and z coordinates from the combinations into a single tensor.
        xyzs = torch.stack(
            (combinations[:, 0], combinations[:, 1], combinations[:, 2]), dim=1
        )

        # Compute takeoff angles using the velocity model and the z coordinate from combinations.
        ihs = takeoff(self.v, self.d, depis, combinations[:, 2])

        # Stack the focal mechanism parameters (strike, dip, rake) from the combinations.
        strikediprakes = torch.stack(
            (combinations[:, 3], combinations[:, 4], combinations[:, 5]), dim=1
        )

        # Compute the amplitudes (cp) and ignore the other two outputs.
        cp, _, _ = radpar(azs, strikediprakes, ihs)

        # Apply preprocessing to the spatial coordinates.
        xyzs_scaled = self.preprocess_transform_xyz(xyzs)

        # Compute polarities as the sign of the amplitudes.
        polarities = torch.sign(cp)

        # Return a dictionary of computed outputs, cast to the target data type.
        return dict(
            XYZ=xyzs_scaled.to(self.torch_dtype),
            P=probs.to(self.torch_dtype),
            az=azs.to(self.torch_dtype),
            ih=ihs.to(self.torch_dtype),
            Cp=cp.to(self.torch_dtype),
            SDR=strikediprakes.to(self.torch_dtype),
            Polarities=polarities.to(self.torch_dtype),
            Amplitudes=cp.abs().to(self.torch_dtype),
        )

    def gen_data_iter_utils_xyz_point(self, start_idx, batch_size, total_combinations):
        end_idx = min(start_idx + batch_size, total_combinations)
        # Genera indici per ogni dimensione
        idx_flattened = torch.arange(start_idx, end_idx)
        
        # Calcola gli indici multidimensionali
        n_xyz = len(self.xyz_points)
        n_strike = len(self.strike_grid)
        n_dip = len(self.dip_grid)
        n_rake = len(self.rake_grid)
        
        idx_xyz = (idx_flattened // (n_strike * n_dip * n_rake)) % n_xyz
        idx_strike = (idx_flattened // (n_dip * n_rake)) % n_strike
        idx_dip = (idx_flattened // n_rake) % n_dip
        idx_rake = idx_flattened % n_rake
        
        # Seleziona i valori corrispondenti
        xyzs = self.xyz_points[idx_xyz]
        strikes = self.strike_grid[idx_strike]
        dips = self.dip_grid[idx_dip]
        rakes = self.rake_grid[idx_rake]
        
        xs, ys, zs = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
        return xs, ys, zs, strikes, dips, rakes

    # --- Batch data generation utilities ---
    def gen_data_iter_utils(
        self, start_idx: int, batch_size: int, total_combinations: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generates a batch of grid parameters given flattened indices.

        :param start_idx: Starting index in the flattened combination space
        :param batch_size: Number of combinations to process in this batch
        :param total_combinations: Total number of parameter combinations
        :return: Tuple of tensors (xs, ys, zs, strikes, dips, rakes)
        """
        # Compute end index without exceeding total combinations
        end_idx = min(start_idx + batch_size, total_combinations)

        # Create a range of flattened indices for this batch
        idx_flattened = torch.arange(start_idx, end_idx)

        # Grid sizes for each spatial and mechanism parameter
        n_x = len(self.x_grid)
        n_y = len(self.y_grid)
        n_z = len(self.z_grid)
        n_strike = len(self.strike_grid)
        n_dip = len(self.dip_grid)
        n_rake = len(self.rake_grid)

        # Unflatten indices into multi-dimensional indices
        idx_x = (idx_flattened // (n_y * n_z * n_strike * n_dip * n_rake)) % n_x
        idx_y = (idx_flattened // (n_z * n_strike * n_dip * n_rake)) % n_y
        idx_z = (idx_flattened // (n_strike * n_dip * n_rake)) % n_z
        idx_strike = (idx_flattened // (n_dip * n_rake)) % n_strike
        idx_dip = (idx_flattened // n_rake) % n_dip
        idx_rake = idx_flattened % n_rake

        # Index each grid to get parameter values
        xs = self.x_grid[idx_x]
        ys = self.y_grid[idx_y]
        zs = self.z_grid[idx_z]
        strikes = self.strike_grid[idx_strike]
        dips = self.dip_grid[idx_dip]
        rakes = self.rake_grid[idx_rake]
        return xs, ys, zs, strikes, dips, rakes

    # --- Main data iterator yielding batches ---
    def gen_data_iter(
        self, batch_size: int = None
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yields dictionaries of preprocessed data for each batch. Includes spatial coords,
        focal mechanism parameters, travel-time azimuth/depth, calculated polarities, etc.

        :param batch_size: Number of samples per batch (defaults to self.batch_size)
        :yield: dict with keys ["XYZ", "P", "az", "ih", "Cp", "SDR", "Polarities", "Amplitudes"]
        """
        # Polygon-based sampling not supported here
        if self.polygon is not None:
            raise NotImplementedError("Cannot generate data with polygon")

        # Use default batch size if none provided
        if batch_size is None:
            batch_size = self.batch_size

        # Compute total number of parameter combinations
        if self.xyz_points is not None:
            total_combinations = (
                len(self.xyz_points) * 
                len(self.strike_grid) * 
                len(self.dip_grid) * 
                len(self.rake_grid)
            )
            function_utils = self.gen_data_iter_utils_xyz_point
        else:
            total_combinations = (
                len(self.x_grid) * 
                len(self.y_grid) * 
                len(self.z_grid) * 
                len(self.strike_grid) * 
                len(self.dip_grid) * 
                len(self.rake_grid)
            )
            function_utils = self.gen_data_iter_utils

        # Iterate in steps of batch_size
        for start_idx in range(0, total_combinations, batch_size):
            # Retrieve parameter grids for this batch
            xs, ys, zs, strikes, dips, rakes = function_utils(
                start_idx, batch_size, total_combinations
            )

            # Stack spatial coordinates
            xyzs = torch.stack((xs, ys, zs), dim=1)

            # Compute azimuths from station to each sample point
            azs = azimuth(self.xstaz, self.ystaz, xs, ys)
            # Compute hypocentral distances based on horizontal offsets
            depis = torch.sqrt(
                (xs[:, None] - self.xstaz[None, :]) ** 2
                + (ys[:, None] - self.ystaz[None, :]) ** 2
            ).to(dtype=self.torch_dtype)
            # Compute detection probabilities from distances
            probs = compute_probs(depis)

            # Stack focal mechanism parameters
            strikediprakes = torch.stack((strikes, dips, rakes), dim=1)
            # Compute takeoff angles for each phase
            ihs = takeoff(self.v, self.d, depis, zs)

            # Compute radiation pattern (polarity and amplitude)
            cp, _, _ = radpar(azs, strikediprakes, ihs)

            # Preprocess spatial coordinates (e.g., normalization)
            xyzs_scaled = self.preprocess_transform_xyz(xyzs)

            # Determine polarity signs and absolute amplitudes
            polarities = torch.sign(cp)
            batch_data = dict(
                XYZ=xyzs_scaled.to(self.torch_dtype),
                P=probs.to(self.torch_dtype),
                az=azs.to(self.torch_dtype),
                ih=ihs.to(self.torch_dtype),
                Cp=cp.to(self.torch_dtype),
                SDR=strikediprakes.to(self.torch_dtype),
                Polarities=polarities.to(self.torch_dtype),
                Amplitudes=cp.abs().to(self.torch_dtype),
            )

            yield batch_data

    def preprocess_transform_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Transform the input xyz coordinates by scaling them into a new range.

        If a scaling range is defined (i.e., self.xyz_scaling_range is not None), this function
        first normalizes the coordinates based on the original boundaries (self.xyz_min to self.xyz_max)
        and then scales them to fit within the desired scaled range (self.xyz_scaled_min to self.xyz_scaled_max).
        If no scaling range is defined, the original xyz values are returned.

        Args:
            xyz (torch.Tensor): Input tensor containing xyz coordinates.

        Returns:
            torch.Tensor: Transformed xyz coordinates.
        """
        # If no scaling range is provided, return the original coordinates.
        if self.xyz_scaling_range is None:
            return xyz

        # Normalize the coordinates: shift by the minimum and divide by the total range.
        xyz_std = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        # Scale the normalized coordinates to the desired range.
        return (
            xyz_std * (self.xyz_scaled_max - self.xyz_scaled_min) + self.xyz_scaled_min
        )

    def preprocess_inverse_transform_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse transformation to the scaled xyz coordinates.

        This function reverses the transformation applied by preprocess_transform_xyz.
        If no scaling range is defined, the input xyz values are returned unchanged.

        Args:
            xyz (torch.Tensor): Transformed xyz coordinates.

        Returns:
            torch.Tensor: Original xyz coordinates restored.
        """
        # If no scaling range is provided, return the original coordinates.
        if self.xyz_scaling_range is None:
            return xyz

        # Normalize the scaled coordinates to a 0-1 range based on the scaled boundaries.
        xyz_std = (xyz - self.xyz_scaled_min) / (
            self.xyz_scaled_max - self.xyz_scaled_min
        )
        # Scale the normalized values back to the original coordinate range.
        return xyz_std * (self.xyz_max - self.xyz_min) + self.xyz_min

    def preprocess_transform(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Apply preprocessing transformations to a dictionary of data.

        This function checks each key in the input dictionary. If the key is 'xyz',
        it applies the spatial transformation defined by preprocess_transform_xyz.
        For other tensor values, it simply clones them; non-tensor values are passed through as-is.

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing data tensors to be transformed.

        Returns:
            dict[str, torch.Tensor]: Dictionary with preprocessed data.
        """
        data_out = dict()
        for key, value in data.items():
            if key == "xyz":
                # Apply the spatial transformation to the 'xyz' key.
                v = self.preprocess_transform_xyz(value)
            elif isinstance(value, torch.Tensor):
                # For other tensor values, clone to avoid modifying the original.
                v = value.clone()
            else:
                # For non-tensor values, simply pass them through.
                v = value
            data_out[key] = v

        return data_out

    def preprocess_inverse_transform(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Apply the inverse preprocessing transformation to a dictionary of data.

        For the 'xyz' key, this function applies the inverse spatial transformation defined by
        preprocess_inverse_transform_xyz. For all other tensor values, it clones them.

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing transformed data tensors.

        Returns:
            dict[str, torch.Tensor]: Dictionary with data tensors reverted to their original space.
        """

        data_out = dict()
        for key, value in data.items():
            if key == "xyz":
                # Apply the inverse transformation to the 'xyz' key.
                v = self.preprocess_inverse_transform_xyz(value)
            else:
                # For other tensor values, clone them to ensure no in-place modifications.
                v = value.clone()
            data_out[key] = v

        return data_out

    def read_test_dat(self, test_data_path: Path) -> dict:
        """
        Read and parse the test data file containing event information.

        The test data file is expected to have a header line that contains the epicenter
        information (latitude in degrees and minutes, longitude in degrees and minutes, and depth)
        followed by lines for each station event with parameters:
            - azimuth (az)
            - takeoff angle (ih)
            - amplitudes value (cp)
            - station name

        The header is parsed character by character to extract:
            - Latitude: degrees and minutes (converted to decimal degrees)
            - Longitude: degrees and minutes (converted to decimal degrees)
            - Depth (as a float)

        Using the extracted epicenter information, the epicentral coordinates (x, y, z) are computed
        via the `gclc` function. Then, for each event line, event parameters are collected and sorted
        based on the station names. The station-specific event parameters are then assigned to their
        corresponding stations. Finally, theoretical azimuth and takeoff angles are computed and all the
        information is returned in a dictionary.

        Args:
            test_data_path (Path): Path to the test data file.

        Returns:
            dict: A dictionary containing:
                - 'stations': List of station names (sorted).
                - 'stations_mask': A boolean tensor mask indicating which of self.stations are present in the test data.
                - 'batch': A dictionary with the following keys:
                    - 'xyz': Tensor of epicentral coordinates [x, y, z].
                    - 'az': Tensor of measured azimuth values for each station.
                    - 'ih': Tensor of measured takeoff angles for each station.
                    - 'cp': Tensor of measured amplitudes values.
                    - 'az_th': Theoretical azimuth computed from epicenter to stations.
                    - 'ih_th': Theoretical takeoff angles computed using the velocity model.
        """
        # Read all lines from the file and strip any extra whitespace.
        with open(test_data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # The first line is the header that contains the epicenter information.
        header = lines[0]

        # ---- Parse the header to extract latitude, longitude, and depth ----
        # Skip leading spaces.
        i_cur = 0
        while header[i_cur] == " ":
            i_cur += 1

        # Extract latitude degrees until a '-' is encountered.
        lat_deg_s = ""
        while header[i_cur] != "-":
            lat_deg_s += header[i_cur]
            i_cur += 1
        # Skip spaces following the '-' separator.
        while header[i_cur] == " ":
            i_cur += 1
        # Convert the string to an integer.
        lat_deg = int(lat_deg_s)

        # Extract the latitude minutes.
        lat_min_s = ""
        i_cur += 1  # Skip the '-' separator.
        while header[i_cur] != " ":
            lat_min_s += header[i_cur]
            i_cur += 1
        lat_min = float(lat_min_s)

        # Skip spaces before longitude.
        while header[i_cur] == " ":
            i_cur += 1

        # Extract longitude degrees until a '-' is encountered.
        lon_deg_s = ""
        while header[i_cur] != "-":
            lon_deg_s += header[i_cur]
            i_cur += 1
        lon_deg = int(lon_deg_s)

        # Skip spaces following the '-' separator.
        while header[i_cur] == " ":
            i_cur += 1

        # Extract the longitude minutes.
        lon_min_s = ""
        i_cur += 1  # Skip the '-' separator.
        while header[i_cur] != " ":
            lon_min_s += header[i_cur]
            i_cur += 1
        lon_min = float(lon_min_s)

        # Skip any remaining spaces and extract the depth value.
        while header[i_cur] == " ":
            i_cur += 1
        depth = float(header[i_cur:].strip())

        # Convert latitude and longitude to decimal degrees.
        latitude = lat_deg + lat_min / 60
        longitude = lon_deg + lon_min / 60

        # Retrieve reference coordinates from the instance.
        latref = self.latref
        lonref = self.lonref

        # Convert the epicenter from geographic to local coordinates using gclc.
        yepic, xepic = gclc(
            latref,
            lonref,
            torch.tensor([latitude], dtype=self.torch_dtype),
            torch.tensor([longitude], dtype=self.torch_dtype),
        )
        xepic = xepic.to(self.device)
        yepic = yepic.to(self.device)
        zepic = torch.tensor([depth], dtype=self.torch_dtype).to(self.device)

        # Stack the epicenter coordinates into a single tensor [x, y, z].
        xyz_epic = torch.stack((xepic, yepic, zepic), dim=1)

        # ---- Parse event data lines (starting from line 1) ----
        ev_azs_l = []  # List for event azimuths.
        ev_ihs_l = []  # List for event takeoff angles.
        ev_cps_l = []  # List for event amplitudes values.
        ev_stations = []  # List for station names corresponding to the events.

        for line in lines[1:]:
            # Split each line into its components.
            event = line.strip().split()
            az = float(event[0])
            ih = float(event[1])
            cp = float(event[2])
            station = event[3]

            ev_azs_l.append(az)
            ev_ihs_l.append(ih)
            ev_cps_l.append(cp)
            ev_stations.append(station)

        # ---- Sort event data based on station names ----
        # Compute sorted indices for the event stations.
        idxsort_stations = torch.tensor(np.argsort(ev_stations))

        # Sort each event list using the sorted indices.
        ev_stations = [ev_stations[i] for i in idxsort_stations]
        ev_azs_l = [ev_azs_l[i] for i in idxsort_stations]
        ev_ihs_l = [ev_ihs_l[i] for i in idxsort_stations]
        ev_cps_l = [ev_cps_l[i] for i in idxsort_stations]

        # Convert event lists to tensors.
        ev_azs = torch.tensor(ev_azs_l, dtype=self.torch_dtype)
        ev_ihs = torch.tensor(ev_ihs_l, dtype=self.torch_dtype)
        ev_cps = torch.tensor(ev_cps_l, dtype=self.torch_dtype)

        # ---- Create a mask for stations present in the data ----
        stations_mask = torch.tensor(
            [station in ev_stations for station in self.stations]
        )
        # Initialize tensors for station-specific event parameters with zeros.
        azs = torch.zeros((1, len(self.stations)), dtype=self.torch_dtype)
        ihs = torch.zeros((1, len(self.stations)), dtype=self.torch_dtype)
        cps = torch.zeros((1, len(self.stations)), dtype=self.torch_dtype)

        # Fill in the event parameters for the stations indicated by the mask.
        azs[0, stations_mask] = ev_azs
        ihs[0, stations_mask] = ev_ihs
        cps[0, stations_mask] = ev_cps

        # ---- Compute theoretical values for azimuth and takeoff angles ----
        # Compute theoretical azimuth from epicenter to stations.
        azs_th = azimuth(
            self.xstaz, self.ystaz, xepic.to(self.device), yepic.to(self.device)
        ).cpu()

        # Compute horizontal distances (depis) from epicenter to stations.
        depis_th = torch.sqrt(
            (xepic[:, None] - self.xstaz[None, :]) ** 2
            + (yepic[:, None] - self.ystaz[None, :]) ** 2
        ).to(dtype=self.torch_dtype)

        # Compute theoretical takeoff angles using the velocity model.
        ihs_th = takeoff(self.v, self.d, depis_th, zepic).cpu()

        # ---- Package the parsed and computed data into a dictionary ----
        test_data = dict(
            stations=ev_stations,  # Sorted list of station names from the test file.
            stations_mask=stations_mask,  # Boolean mask indicating stations with events.
            batch=dict(
                xyz=xyz_epic,  # Epicenter coordinates as a tensor.
                az=azs,  # Measured azimuth values for stations.
                ih=ihs,  # Measured takeoff angles for stations.
                cp=cps,  # Measured amplitude values.
                az_th=azs_th,  # Theoretical azimuth values.
                ih_th=ihs_th,  # Theoretical takeoff angles.
            ),
        )

        return test_data

    def set_seed(self, seed: int):
        """
        Set the random seed for the dataset's random number generator.

        This method ensures reproducibility by manually setting the seed of the generator.

        Args:
            seed (int): The seed value to set.

        Returns:
            self: The instance with the updated random seed.
        """
        self.generator.manual_seed(seed)
        return self

    @staticmethod
    def point_in_polygon(points: torch.Tensor, poly: torch.Tensor) -> torch.BoolTensor:
        """
        Ray-casting algorithm to test if each point lies inside a 2D polygon.

        :param points: Tensor of shape [n_points, 2] containing x,y coordinates to test
        :param poly:   Tensor of shape [n_vertices, 2] defining the polygon in order
        :return:       Boolean tensor of length n_points (True if inside)
        """
        # Roll the polygon vertices by one to align each vertex j with its predecessor i

        poly_roll = torch.roll(poly, shifts=1, dims=0)

        # Extract x,y for current vertices (i) and previous vertices (j)
        xi = poly[:, 0].unsqueeze(0)  # shape: [1, n_vertices]
        yi = poly[:, 1].unsqueeze(0)
        xj = poly_roll[:, 0].unsqueeze(0)
        yj = poly_roll[:, 1].unsqueeze(0)

        # Extract x,y for query points, adding a vertex-dimension for broadcasting
        x = points[:, 0].unsqueeze(1)  # shape: [n_points, 1]
        y = points[:, 1].unsqueeze(1)

        # Condition 1: check if the horizontal ray crosses the vertical span of each edge
        cond1 = (yi > y) != (yj > y)

        # Compute the x-coordinate where the horizontal ray at height y intersects each edge
        # Add a tiny epsilon to denominator to avoid division by zero
        x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi

        # Condition 2: check if point's x is to the left of the intersection point
        cond2 = x < x_intersect

        # Count how many edges each point's ray intersects
        intersections = torch.sum(cond1 & cond2, dim=1)

        # A point is inside the polygon if the count of intersections is odd
        return intersections % 2 == 1
