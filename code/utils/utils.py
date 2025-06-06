#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%

import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# %%

# Determine if the script is running in an interactive environment
_interactive_mode = "ipykernel_launcher" in sys.argv[0] or (
    len(sys.argv) == 1 and sys.argv[0] == ""
)


def is_interactive() -> bool:
    """
    Check if the current Python session is interactive.

    Returns:
        bool:   True if running in an interactive environment
                (e.g., Jupyter Notebook, IPython),
                False otherwise.

    The function checks:
    - If 'ipykernel_launcher' is in `sys.argv[0]`, which indicates
        Jupyter Notebook or IPython.
    - If `sys.argv` contains only one element (empty script name),
        indicating an interactive Python shell.
    """
    return _interactive_mode


def myshow(fig=None, plot_show=True):
    """
    Show or close the current plot based on the interactive mode and user preference.
    Args:
        plot_show (bool): If True, display the plot; if False, close it.
    """
    if fig is None:
        if _interactive_mode and plot_show:
            plt.show()
        else:
            plt.close()
    else:
        if _interactive_mode and plot_show:
            fig.show()
        elif isinstance(fig, plt.Figure):
            fig.close()
        elif isinstance(fig, go.Figure):
            pass


# %%


def create_test_dataset(
    df: pd.DataFrame,
    stations: list[str],
    dtype: str = "float32",
    device: str = "cuda",
) -> dict:
    """
    Create a test dataset in the form of PyTorch tensors from a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing seismic event data.
        stations (list[str]): List of station names used to generate column names.
        dtype (str, optional): The desired PyTorch data type (default: "float32").
        device (str, optional): The device to move tensors to (default: "cuda").

    Returns:
        dict: A dictionary containing tensors for various seismic parameters.
    """

    # Define column names for different seismic parameters
    xyz_cols = ["x", "y", "z"]  # Spatial coordinates
    strikediprake_cols = ["strike", "dip", "rake"]  # Focal mechanism angles
    cp_columns = [f"cp_{i}" for i in stations]  # Amplitude with sign columns
    amplitude_columns = [
        f"amplitude_{i}" for i in stations
    ]  # Absolute amplitude values
    pol_columns = [f"pol_{i}" for i in stations]  # Seismic wave polarity
    az_columns = [f"az_{i}" for i in stations]  # Azimuth angles
    ih_columns = [f"ih_{i}" for i in stations]  # Incidence angles

    # Convert the required columns to PyTorch tensors and move them to the specified device
    output = {
        "XYZ": torch.tensor(
            df[xyz_cols].values,
            dtype=getattr(torch, dtype),  # Dynamically assign dtype to the torch module
        ).to(device),
        # Use amplitude columns if present, otherwise fall back to absolute Amplitude with sign values
        "Amplitudes": (
            torch.tensor(df[amplitude_columns].values, dtype=getattr(torch, dtype)).to(
                device
            )
            if np.all(
                [amp in df.columns for amp in amplitude_columns]
            )  # Check if all amplitude columns exist
            else torch.tensor(
                df[cp_columns]
                .abs()
                .values,  # Use absolute Amplitude with sign values if amplitudes are missing
                dtype=getattr(torch, dtype),
            ).to(device)
        ),
        # Use polarity columns if present, otherwise compute sign from Amplitude with sign values
        "Polarities": (
            torch.tensor(df[pol_columns].values, dtype=getattr(torch, dtype)).to(device)
            if np.all(
                [pol in df.columns for pol in pol_columns]
            )  # Check if all polarity columns exist
            else torch.tensor(
                np.sign(
                    df[cp_columns].values
                ),  # Compute polarities using the sign function
                dtype=getattr(torch, dtype),
            ).to(device)
        ),
        # Strike-Dip-Rake values
        "SDR": torch.tensor(
            df[strikediprake_cols].values,
            dtype=getattr(torch, dtype),
        ).to(device),
        # Azimuth values
        "az": torch.tensor(
            df[az_columns].values,
            dtype=getattr(torch, dtype),
        ).to(device),
        # Incidence angle values
        "ih": torch.tensor(
            df[ih_columns].values,
            dtype=getattr(torch, dtype),
        ).to(device),
    }
    return output


def compute_figure_rows_cols(n_plots: int) -> tuple:
    """
    Compute the number of rows and columns needed to arrange a given number of
        plots in a grid.

    Args:
        n_plots (int): The total number of plots to arrange.

    Returns:
        tuple: A tuple (n_rows, n_cols) indicating the optimal grid layout.
    """
    n_cols = math.ceil(
        math.sqrt(n_plots)
    )  # Determine number of columns based on square root
    n_rows = math.ceil(n_plots / n_cols)  # Compute rows to fit all plots
    return n_rows, n_cols


# %%


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    """
    Convert degrees to radians.

    Args:
        deg (torch.Tensor): Angle in degrees.

    Returns:
        torch.Tensor: Angle in radians.
    """
    return deg * (torch.pi / 180)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """
    Convert radians to degrees.

    Args:
        rad (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Angle in degrees.
    """
    return rad * (180 / torch.pi)


def strikedip2norm_inner(strike: torch.Tensor, dip: torch.Tensor) -> torch.Tensor:
    """
    Compute the normal vector to the fault plane from strike and dip angles.

    Args:
        strike (torch.Tensor): Strike angle in degrees.
        dip (torch.Tensor): Dip angle in degrees.

    Returns:
        torch.Tensor: Normal vector (n, e, u) components stacked along the 
                      last dimension.
    """
    strike_rad = deg2rad(strike)
    dip_rad = deg2rad(dip)

    n = -torch.sin(dip_rad) * torch.sin(strike_rad)  # North component
    e = torch.sin(dip_rad) * torch.cos(strike_rad)  # East component
    u = torch.cos(dip_rad)  # Upward component

    return torch.stack([n, e, u], dim=-1)


def sdr2slip_inner(
    strike: torch.Tensor, dip: torch.Tensor, rake: torch.Tensor
) -> torch.Tensor:
    """
    Compute the slip vector from strike, dip, and rake angles.

    Args:
        strike (torch.Tensor): Strike angle in degrees.
        dip (torch.Tensor): Dip angle in degrees.
        rake (torch.Tensor): Rake angle in degrees.

    Returns:
        torch.Tensor: Slip vector (n, e, u) components stacked along the last dimension.
    """
    strike_rad = deg2rad(strike)
    dip_rad = deg2rad(dip)
    rake_rad = deg2rad(rake)

    # Compute the slip components
    n = torch.cos(rake_rad) * torch.cos(strike_rad) + torch.sin(rake_rad) * torch.cos(
        dip_rad
    ) * torch.sin(strike_rad)
    e = torch.cos(rake_rad) * torch.sin(strike_rad) - torch.sin(rake_rad) * torch.cos(
        dip_rad
    ) * torch.cos(strike_rad)
    u = torch.sin(rake_rad) * torch.sin(dip_rad)

    return torch.stack([n, e, u], dim=-1)


def norm2strikedip(n: torch.Tensor, e: torch.Tensor, u: torch.Tensor) -> tuple:
    """
    Convert a normal vector to strike and dip angles.

    Args:
        n (torch.Tensor): North component of the normal vector.
        e (torch.Tensor): East component of the normal vector.
        u (torch.Tensor): Upward component of the normal vector.

    Returns:
        tuple: Strike and dip angles in degrees.
    """
    # Compute strike angle (azimuth of the fault plane)
    strike = torch.remainder(rad2deg(torch.atan2(-n, e)), 360)

    # Compute dip angle (inclination of the fault plane)
    dip = rad2deg(torch.acos(u / torch.sqrt(n**2 + e**2 + u**2)))

    return strike, dip


def normslip2sdr(normal: torch.Tensor, slip: torch.Tensor) -> tuple:
    """
    Convert a normal and slip vector to strike, dip, and rake angles.

    Args:
        normal (torch.Tensor): Normal vector (Nx3 tensor).
        slip (torch.Tensor): Slip vector (Nx3 tensor).

    Returns:
        tuple: Strike, dip, and rake angles in degrees.
    """
    # Compute slip vector magnitude and normalize
    s_mag = torch.sqrt(torch.sum(slip**2, dim=1))
    slip_norm = slip / s_mag[:, None]

    # Convert normal vector to strike and dip angles
    strike, dip = norm2strikedip(normal[:, 0], normal[:, 1], normal[:, 2])

    striker = deg2rad(strike)

    # Compute rake angle
    v = torch.cos(striker) * slip_norm[:, 0] + torch.sin(striker) * slip_norm[:, 1]
    v = torch.clamp(v, -1, 1)  # Ensure values are within valid range for acos

    rake = rad2deg(torch.acos(v))

    # Adjust rake sign based on vertical slip direction
    rake[slip[:, 2] < 0] *= -1

    return strike, dip, rake


def conjugate_torch(strike_dip_rake_torch: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate plane solution for given strike, dip, and rake angles.

    Args:
        strike_dip_rake_torch (torch.Tensor): Tensor of shape (N, 3) containing 
                                              strike, dip, and rake angles.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing conjugate strike, dip, and rake angles.
    """
    assert strike_dip_rake_torch.shape[-1] == 3, "The input must have 3 columns"
    assert len(strike_dip_rake_torch.shape) == 2, "The input must have 2 dimensions"

    # Extract strike, dip, and rake components
    strike, dip, rake = torch.unbind(strike_dip_rake_torch, dim=-1)

    # Compute the normal and slip vectors
    normal = strikedip2norm_inner(strike, dip)
    slip = sdr2slip_inner(strike, dip, rake)

    # Determine the slip direction sign
    signslip = torch.signbit(slip[:, 2:3])

    # Compute the conjugate strike, dip, and rake angles
    strike2, dip2, rake2 = normslip2sdr(
        (1 - 2 * signslip) * slip,  # Flip sign if needed
        (1 - 2 * signslip) * normal,
    )

    return torch.stack((strike2, dip2, rake2), dim=1)


# %%
