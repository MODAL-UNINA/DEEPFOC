#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplstereonet
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import scipy.special as sc
from .PT_axis import FocalMechanism, FocalMechanism_torch
from typing import Tuple, Union, List

# %%


def do_conjugate_sc(
    dd: Union[float, np.ndarray],
    da: Union[float, np.ndarray],
    sa: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Calculates the conjugate plane parameters (strike, dip, rake) using
    spherical trigonometry functions from scipy.special (cosdg, sindg).
    This function likely implements a specific formula for conjugate planes.

    Args:
        dd (Union[float, np.ndarray]): The 'dip direction' of the original plane in degrees.
                                       This is often (strike + 90) % 360.
        da (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        sa (Union[float, np.ndarray]): The slip (rake) angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the 'dip direction', dip angle, and slip (rake) angle
            of the conjugate plane, all in degrees.
    """
    # Convert dip direction to strike for calculations (assuming dd is strike + 90)
    strk = (dd - 90) % 360

    # Store the sign of the original slip angle for later use in rake calculation.
    sgn = sa

    # Numerator for the arctan2 calculation of the new dip direction.
    top = sc.cosdg(sa) * sc.sindg(strk) - sc.cosdg(da) * sc.sindg(sa) * sc.cosdg(strk)
    # Denominator for the arctan2 calculation of the new dip direction.
    bot = sc.cosdg(sa) * sc.cosdg(strk) + sc.cosdg(da) * sc.sindg(sa) * sc.sindg(strk)

    # Calculate the new dip direction (dd2) in degrees.
    dd2 = np.rad2deg(np.arctan2(top, bot))
    # Convert new dip direction to strike for subsequent calculations.
    strk2 = (dd2 - 90) % 360

    # Adjust dd2 based on the sign of the original slip angle,
    # ensuring the dip direction is consistent with convention.
    if np.copysign(1, sa) < 0:
        dd2 -= 180
    # Normalize dd2 to be within [0, 360).
    dd2 %= 360

    # Calculate the new dip angle (da2).
    da2 = np.rad2deg(np.arccos(sc.sindg(np.abs(sa)) * sc.sindg(da)))

    # Calculate an intermediate value related to the new rake angle.
    xlam2 = -sc.cosdg(strk2) * sc.sindg(da) * sc.sindg(strk) + sc.sindg(
        strk2
    ) * sc.sindg(da) * sc.cosdg(strk)

    # Clip xlam2 to avoid issues with arccos due to floating point inaccuracies.
    if np.abs(xlam2) > 1:
        xlam2 = np.copysign(1, xlam2)

    # Calculate the new rake angle in radians, preserving the sign from the original rake.
    xlam2 = np.copysign(np.arccos(xlam2), sgn)
    # Convert the new rake angle to degrees.
    sa2 = np.rad2deg(xlam2)

    return dd2, da2, sa2


def cos(x_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the cosine of an angle given in degrees, with special handling
    for angles that are exact multiples of 90, 180, 270, 360 to avoid floating point errors.

    Args:
        x_deg (Union[float, np.ndarray]): The angle in degrees.

    Returns:
        Union[float, np.ndarray]: The cosine of the angle.
    """
    # Check for angles where cosine is exactly 0 (90, 270 degrees etc.)
    if np.isclose((x_deg - 90) % 180, 0):
        return np.float64(0.0)
    # Check for angles where cosine is exactly 1 (0, 360 degrees etc.)
    if np.isclose(x_deg % 360, 0):
        return np.float64(1.0)
    # Check for angles where cosine is exactly -1 (180 degrees etc.)
    if np.isclose((x_deg - 180) % 360, 0):
        return np.float64(-1.0)
    # For other angles, calculate cosine after converting to radians.
    return np.cos(np.deg2rad(x_deg))


def sin(x_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the sine of an angle given in degrees, with special handling
    for angles that are exact multiples of 90, 180, 270, 360 to avoid floating point errors.

    Args:
        x_deg (Union[float, np.ndarray]): The angle in degrees.

    Returns:
        Union[float, np.ndarray]: The sine of the angle.
    """
    # Check for angles where sine is exactly 0 (0, 180, 360 degrees etc.)
    if np.isclose(x_deg % 180, 0):
        return np.float64(0.0)
    # Check for angles where sine is exactly 1 (90 degrees etc.)
    if np.isclose((x_deg - 90) % 360, 0):
        return np.float64(1.0)
    # Check for angles where sine is exactly -1 (270 degrees etc.)
    if np.isclose((x_deg - 270) % 360, 0):
        return np.float64(-1.0)
    # For other angles, calculate sine after converting to radians.
    return np.sin(np.deg2rad(x_deg))


def do_conjugate_adj(
    dd: Union[float, np.ndarray],
    da: Union[float, np.ndarray],
    sa: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Calculates the conjugate plane parameters (strike, dip, rake) using
    custom `cos` and `sin` functions (do_conjugate_adj).
    This function likely implements a specific formula for conjugate planes.

    Args:
        dd (Union[float, np.ndarray]): The 'dip direction' of the original plane in degrees.
                                       This is often (strike + 90) % 360.
        da (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        sa (Union[float, np.ndarray]): The slip (rake) angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the 'dip direction', dip angle, and slip (rake) angle
            of the conjugate plane, all in degrees.
    """
    # Convert dip direction to strike for calculations (assuming dd is strike + 90).
    strk = (dd - 90) % 360
    # Store the sign of the original slip angle for later use in rake calculation.
    sgn = sa

    # Numerator for the arctan2 calculation of the new dip direction using custom sin/cos.
    top = cos(sa) * sin(strk) - cos(da) * sin(sa) * cos(strk)
    # Denominator for the arctan2 calculation of the new dip direction using custom sin/cos.
    bot = cos(sa) * cos(strk) + cos(da) * sin(sa) * sin(strk)

    # Calculate the new dip direction (dd2) in degrees.
    dd2 = np.rad2deg(np.arctan2(top, bot))
    # Convert new dip direction to strike for subsequent calculations.
    strk2 = (dd2 - 90) % 360

    # Adjust dd2 based on the sign of the original slip angle,
    # ensuring the dip direction is consistent with convention.
    if np.copysign(1, sa) < 0:
        dd2 -= 180
    # Normalize dd2 to be within [0, 360).
    dd2 %= 360

    # Calculate the new dip angle (da2) using custom sin.
    da2 = np.rad2deg(np.arccos(sin(np.abs(sa)) * sin(da)))

    # Calculate an intermediate value related to the new rake angle using custom sin/cos.
    xlam2 = -cos(strk2) * sin(da) * sin(strk) + sin(strk2) * sin(da) * cos(strk)

    # Clip xlam2 to avoid issues with arccos due to floating point inaccuracies.
    if np.abs(xlam2) > 1:
        xlam2 = np.copysign(1, xlam2)

    # Calculate the new rake angle in radians, preserving the sign from the original rake.
    xlam2 = np.copysign(np.arccos(xlam2), sgn)
    # Convert the new rake angle to degrees.
    sa2 = np.rad2deg(xlam2)

    return dd2, da2, sa2


def do_conjugate(
    dd: Union[float, np.ndarray],
    da: Union[float, np.ndarray],
    sa: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Calculates the conjugate plane parameters (strike, dip, rake) using
    standard numpy trigonometric functions (np.sin, np.cos).
    This function likely implements a specific formula for conjugate planes.

    Args:
        dd (Union[float, np.ndarray]): The 'dip direction' of the original plane in degrees.
                                       This is often (strike + 90) % 360.
        da (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        sa (Union[float, np.ndarray]): The slip (rake) angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the 'dip direction', dip angle, and slip (rake) angle
            of the conjugate plane, all in degrees.
    """
    # Convert dip direction to strike angle for calculations.
    phi1 = (dd - 90) % 360

    # Convert all input angles to radians.
    phi1 = np.deg2rad(phi1)
    del1 = np.deg2rad(da)
    sgn = sa  # Store the sign of the original slip angle
    xlam1 = np.deg2rad(sa)

    # Numerator for the arctan2 calculation of the new dip direction.
    top = np.cos(xlam1) * np.sin(phi1) - np.cos(del1) * np.sin(xlam1) * np.cos(phi1)
    # Denominator for the arctan2 calculation of the new dip direction.
    bot = np.cos(xlam1) * np.cos(phi1) + np.cos(del1) * np.sin(xlam1) * np.sin(phi1)

    # Calculate the new dip direction (dd2) in degrees.
    dd2 = np.rad2deg(np.arctan2(top, bot))
    # Convert new dip direction to strike for subsequent calculations.
    phi2 = np.deg2rad(dd2 - 90)

    # Adjust dd2 based on the sign of the original slip angle.
    if sa < 0:
        dd2 -= 180
    # Normalize dd2 to be within [0, 360).
    dd2 %= 360

    # Calculate the new dip angle (da2).
    da2 = np.rad2deg(np.arccos(np.sin(np.abs(xlam1)) * np.sin(del1)))

    # Calculate an intermediate value related to the new rake angle.
    xlam2 = -np.cos(phi2) * np.sin(del1) * np.sin(phi1) + np.sin(phi2) * np.sin(
        del1
    ) * np.cos(phi1)

    # Clip xlam2 to avoid issues with arccos due to floating point inaccuracies.
    if np.abs(xlam2) > 1:
        xlam2 = np.copysign(1, xlam2)

    # Calculate the new rake angle in radians, preserving the sign from the original rake.
    xlam2 = np.copysign(np.arccos(xlam2), sgn)
    # Convert the new rake angle to degrees.
    sa2 = np.rad2deg(xlam2)

    return dd2, da2, sa2


def conjugate(
    strike: Union[float, np.ndarray],
    dip: Union[float, np.ndarray],
    rake: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Computes the conjugate plane for a given fault plane (strike, dip, rake)
    using the `do_conjugate` function, which relies on standard numpy trigonometric functions.

    Args:
        strike (Union[float, np.ndarray]): The strike angle of the original plane in degrees.
        dip (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        rake (Union[float, np.ndarray]): The rake angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the strike, dip, and rake angles of the conjugate plane, all in degrees.
    """
    # Convert strike to dip direction (strike + 90) as expected by do_conjugate.
    dd, da, sa = (strike + 90) % 360, dip, rake
    # Ensure inputs are float64 for consistent calculations.
    dd = np.float64(dd)
    da = np.float64(da)
    sa = np.float64(sa)
    # Call the core conjugate calculation function.
    dd2, da2, sa2 = do_conjugate(dd, da, sa)
    # Convert dip direction back to strike angle (dd2 - 90) and normalize.
    return (dd2 - 90) % 360, da2, sa2


def conjugate_adj(
    strike: Union[float, np.ndarray],
    dip: Union[float, np.ndarray],
    rake: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Computes the conjugate plane for a given fault plane (strike, dip, rake)
    using the `do_conjugate_adj` function, which relies on custom trigonometric functions
    with float precision handling.

    Args:
        strike (Union[float, np.ndarray]): The strike angle of the original plane in degrees.
        dip (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        rake (Union[float, np.ndarray]): The rake angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the strike, dip, and rake angles of the conjugate plane, all in degrees.
    """
    # Convert strike to dip direction (strike + 90) as expected by do_conjugate_adj.
    dd, da, sa = (strike + 90) % 360, dip, rake
    # Call the core conjugate calculation function with custom trig.
    dd2, da2, sa2 = do_conjugate_adj(dd, da, sa)
    # Convert dip direction back to strike angle (dd2 - 90) and normalize.
    return (dd2 - 90) % 360, da2, sa2


def conjugate_sc(
    strike: Union[float, np.ndarray],
    dip: Union[float, np.ndarray],
    rake: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """
    Computes the conjugate plane for a given fault plane (strike, dip, rake)
    using the `do_conjugate_sc` function, which relies on `scipy.special`
    trigonometric functions.

    Args:
        strike (Union[float, np.ndarray]): The strike angle of the original plane in degrees.
        dip (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        rake (Union[float, np.ndarray]): The rake angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the strike, dip, and rake angles of the conjugate plane, all in degrees.
    """
    # Convert strike to dip direction (strike + 90) as expected by do_conjugate_sc.
    dd, da, sa = (strike + 90) % 360, dip, rake
    # Call the core conjugate calculation function with scipy.special trig.
    dd2, da2, sa2 = do_conjugate_sc(dd, da, sa)
    # Convert dip direction back to strike angle (dd2 - 90) and normalize.
    return (dd2 - 90) % 360, da2, sa2


def conjugates(
    strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes conjugate planes for arrays of strike, dip, and rake angles
    using the `conjugate` function (standard numpy trigonometric functions).

    Args:
        strikes (np.ndarray): An array of strike angles in degrees.
        dips (np.ndarray): An array of dip angles in degrees.
        rakes (np.ndarray): An array of rake angles in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing arrays of the conjugate strike, dip, and rake angles, all in degrees.
    """
    strikes_conj = []
    dips_conj = []
    rakes_conj = []
    # Iterate through each set of strike, dip, rake.
    for strike, dip, rake in zip(strikes, dips, rakes):
        # Calculate the conjugate for the current plane.
        strike_conj, dip_conj, rake_conj = conjugate(strike, dip, rake)
        # Append results to respective lists.
        strikes_conj.append(strike_conj)
        dips_conj.append(dip_conj)
        rakes_conj.append(rake_conj)

    # Convert lists of results to numpy arrays.
    return np.array(strikes_conj), np.array(dips_conj), np.array(rakes_conj)


def conjugates_adj(
    strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes conjugate planes for arrays of strike, dip, and rake angles
    using the `conjugate_adj` function (custom trigonometric functions).

    Args:
        strikes (np.ndarray): An array of strike angles in degrees.
        dips (np.ndarray): An array of dip angles in degrees.
        rakes (np.ndarray): An array of rake angles in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing arrays of the conjugate strike, dip, and rake angles, all in degrees.
    """
    strikes_conj = []
    dips_conj = []
    rakes_conj = []
    # Iterate through each set of strike, dip, rake.
    for strike, dip, rake in zip(strikes, dips, rakes):
        # Calculate the conjugate for the current plane using custom trig.
        strike_conj, dip_conj, rake_conj = conjugate_adj(strike, dip, rake)
        # Append results to respective lists.
        strikes_conj.append(strike_conj)
        dips_conj.append(dip_conj)
        rakes_conj.append(rake_conj)

    # Convert lists of results to numpy arrays.
    return np.array(strikes_conj), np.array(dips_conj), np.array(rakes_conj)


def conjugates_sc(
    strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes conjugate planes for arrays of strike, dip, and rake angles
    using the `conjugate_sc` function (scipy.special trigonometric functions).

    Args:
        strikes (np.ndarray): An array of strike angles in degrees.
        dips (np.ndarray): An array of dip angles in degrees.
        rakes (np.ndarray): An array of rake angles in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing arrays of the conjugate strike, dip, and rake angles, all in degrees.
    """
    strikes_conj = []
    dips_conj = []
    rakes_conj = []
    # Iterate through each set of strike, dip, rake.
    for strike, dip, rake in zip(strikes, dips, rakes):
        # Calculate the conjugate for the current plane using scipy.special trig.
        strike_conj, dip_conj, rake_conj = conjugate_sc(strike, dip, rake)
        # Append results to respective lists.
        strikes_conj.append(strike_conj)
        dips_conj.append(dip_conj)
        rakes_conj.append(rake_conj)

    # Convert lists of results to numpy arrays.
    return np.array(strikes_conj), np.array(dips_conj), np.array(rakes_conj)


def conjugates_parallel(
    *strikes_dips_rakes: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_jobs: Union[int, None] = None,
    use_adj: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Computes conjugate planes in parallel using joblib.
    Can accept either three separate numpy arrays (strikes, dips, rakes)
    or a single numpy array with three columns ([strike, dip, rake]).

    Args:
        *strikes_dips_rakes (Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
            Variable argument list.
            If 3 arrays: strikes (np.ndarray), dips (np.ndarray), rakes (np.ndarray).
            If 1 array: a 2D numpy array of shape (N, 3) where each row is [strike, dip, rake].
        n_jobs (Union[int, None], optional): Number of parallel jobs to run.
                                             If -1, uses all available CPU cores.
                                             If None, uses os.cpu_count(). Defaults to None.
        use_adj (bool, optional): If True, uses the `conjugate_adj` function (custom trig).
                                  Otherwise, uses the `conjugate` function (standard numpy trig).
                                  Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
            If input was three separate arrays, returns a tuple of three numpy arrays
            (conjugate strikes, dips, rakes).
            If input was a single 2D numpy array, returns a single 2D numpy array
            of conjugate planes.
    """
    # Determine which conjugate function to use based on 'use_adj' flag.
    if use_adj:
        conjugates_func = conjugates_adj
        conjugate_func = conjugate_adj
    else:
        conjugates_func = conjugates
        conjugate_func = conjugate

    # Handle input format: three separate arrays.
    if len(strikes_dips_rakes) == 3:
        strikes, dips, rakes = strikes_dips_rakes
        # Assert that all input arrays have the same length.
        assert (
            len(strikes) == len(dips) == len(rakes)
        ), "All input arrays must have the same length."

        # Process in parallel using tqdm for progress indication.
        strikes_dips_rakes_conj = Parallel(n_jobs=-1)(
            delayed(conjugates_func)(
                strike, dip, rake
            )
            for strike, dip, rake in tqdm(zip(strikes, dips, rakes))
        )
        # Unzip the results into separate strike, dip, rake arrays.
        return zip(*strikes_dips_rakes_conj)
    # Handle input format: a single 2D numpy array.
    else:
        assert (
            len(strikes_dips_rakes) == 1
        ), "Input must be a tuple with a single numpy array or three numpy arrays."
        strikes_dips_rakes = strikes_dips_rakes[0]
        # Assert that the input is a numpy array.
        assert isinstance(
            strikes_dips_rakes, np.ndarray
        ), "Input must be a numpy array."
        # Assert that the input array has 3 columns (strike, dip, rake).
        assert strikes_dips_rakes.shape[1] == 3, "Input array must have 3 columns."

        def conjugate_(batch: np.ndarray) -> np.ndarray:
            """Helper function to apply conjugate_func to a batch of planes."""
            return np.stack(
                [conjugate_func(*strikediprake) for strikediprake in batch], axis=0
            )

        # Determine the number of jobs for parallel processing.
        if n_jobs is None:
            n_jobs = os.cpu_count()  # Use all available CPU cores by default.

        # Split the input array into batches for parallel processing.
        strikes_dips_rakes_batch = np.array_split(strikes_dips_rakes, n_jobs, axis=0)

        # Execute conjugate_ on each batch in parallel.
        strikes_dips_rakes_conj = Parallel(n_jobs=n_jobs)(
            delayed(conjugate_)(batch) for batch in tqdm(strikes_dips_rakes_batch)
        )

        # Concatenate the results from all batches.
        return np.concatenate(strikes_dips_rakes_conj, axis=0)


# %%


def adjust_text_positions(ax: plt.Axes, texts: List[plt.Text], width: float = 0.02, height: float = 0.02):
    """
    Attempt to adjust text positions to minimize overlaps on a matplotlib axis.
    This function iteratively nudges text objects that are overlapping.

    Args:
        ax (plt.Axes): The matplotlib Axes object on which the texts are plotted.
        texts (List[plt.Text]): A list of matplotlib Text objects to adjust.
        width (float, optional): Estimated width of a text object in axis coordinates. Defaults to 0.02.
        height (float, optional): Estimated height of a text object in axis coordinates. Defaults to 0.02.
    """
    # Iterate a fixed number of times to try and resolve overlaps.
    # This prevents infinite loops in complex overlap scenarios.
    for _ in range(100):
        # For each text object, check for overlaps with every other text object.
        for text in texts:
            tx, ty = text.get_position() # Get the current position of the text.
            for other_text in texts:
                # Skip comparison with itself.
                if other_text == text:
                    continue
                ox, oy = other_text.get_position() # Get the position of the other text.
                
                # Check if the bounding boxes of the two text objects overlap.
                # This is a simplified check based on estimated text dimensions.
                if (abs(tx - ox) < width) and (abs(ty - oy) < height):
                    # If they overlap, adjust the position of the current text to move it away.
                    # Determine the direction of adjustment based on the relative positions.
                    direction = np.sign((tx - ox, ty - oy))
                    # Nudge the text by half of its estimated width/height in the determined direction.
                    text.set_position(
                        (
                            tx + direction[0] * width * 0.5,
                            ty + direction[1] * height * 0.5,
                        )
                    )


def plot_focal_mechanisms_pt(
    df_focalmechanism: pd.DataFrame,
    df_stations: pd.DataFrame | None = None,
    linewidth: float = 1.0,
    alpha: float = 0.25,
    line_style: str = "r-",
    auxiliary_line_style: str = "b-",
    palette: str = "Reds",
    vmin: float = -0.2,
    vmax: float = 1,
    figsize: Tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    use_color: bool = False,
    jacknife: bool = False,
    discrepancy_error: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots focal mechanisms (double-couple solutions), P and T axes, and optionally station polarities
    and amplitudes on an equal-area stereonet. This function includes handling for multiple focal mechanisms
    (e.g., from bootstrap or jackknife analyses).

    Args:
        df_focalmechanism (pd.DataFrame): DataFrame with focal mechanism parameters,
                                          must contain 'strike', 'dip', 'rake' columns.
        df_stations (pd.DataFrame | None, optional): DataFrame with station data,
                                                     can contain 'station', 'azimuth', 'takeoff',
                                                     'polarity', and optionally 'amplitudes' columns.
                                                     Defaults to None.
        linewidth (float, optional): Line width for the primary nodal planes. Defaults to 1.0.
        alpha (float, optional): Transparency for the nodal planes. Defaults to 0.25.
        line_style (str, optional): Matplotlib line style for the primary nodal planes (e.g., 'r-'). Defaults to "r-".
        auxiliary_line_style (str, optional): Matplotlib line style for the conjugate nodal planes (e.g., 'b-'). Defaults to "b-".
        palette (str, optional): Matplotlib colormap name for station amplitude colors if use_color is True. Defaults to "Reds".
        vmin (float, optional): Minimum value for colormap normalization. Defaults to -0.2.
        vmax (float, optional): Maximum value for colormap normalization. Defaults to 1.
        figsize (Tuple[int, int] | None, optional): Figure size (width, height) in inches. Defaults to None.
        ax (plt.Axes | None, optional): Pre-existing matplotlib Axes object to plot on.
                                        If None, a new stereonet axis is created. Defaults to None.
        title (str | None, optional): Title for the plot. Defaults to None.
        use_color (bool, optional): If True, colors station polarities/amplitudes based on 'amplitudes' values. Defaults to False.
        jacknife (bool, optional): If True, indicates that focal mechanisms are from a jackknife analysis,
                                   which affects alpha values and P/T axis plotting. Defaults to False.
        discrepancy_error (bool, optional): If True, asserts that df_stations is provided for discrepancy error calculation.
                                            (Note: The calculation itself is not explicitly shown in this snippet).
                                            Defaults to False.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects with the plot.
    """
    # Extract strike, dip, and rake values from the focal mechanism DataFrame.
    strikes = df_focalmechanism["strike"].values
    dips = df_focalmechanism["dip"].values
    rakes = df_focalmechanism["rake"].values

    # Calculate the conjugate planes for the given focal mechanisms.
    strikes_conj, dips_conj, rakes_conj = conjugates(strikes, dips, rakes)

    # If station data is provided, extract relevant columns.
    if df_stations is not None:
        statns = df_stations["station"].values
        azimuths = df_stations["azimuth"].values
        takeoffs = df_stations["takeoff"].values
        polarities = df_stations["polarity"].values
        # 'amplitudes' is optional.
        amp_values = (
            df_stations["amplitudes"].values if "amplitudes" in df_stations else None
        )

    # Create a new stereonet axis if none is provided.
    if ax is None:
        fig, ax = mplstereonet.subplots(projection="equal_area", figsize=figsize)
    else:
        # If an axis is provided, get its figure.
        fig = ax.get_figure()

    # Ensure ax is not None before proceeding.
    assert ax is not None
    Ps = [] # List to store P-axes.
    Ts = [] # List to store T-axes.

    # Determine alpha values for plotting nodal planes based on the number of mechanisms
    # and whether it's a jackknife analysis.
    if len(strikes) > 1 and not jacknife:
        alphas = np.linspace(1.0, 0.3, len(strikes)) # Varying transparency for multiple mechanisms.
    elif jacknife:
        alphas = np.ones(len(strikes)) # Uniform transparency for jackknife results.
    else:
        alphas = [alpha] # Single transparency value for a single mechanism.
        
    # Plotting primary nodal planes and calculating P and T axes.
    for strike, dip, rake, alph in zip(strikes, dips, rakes, alphas):
        ax.plane(strike, dip, line_style, linewidth=linewidth, alpha=alph)
        # Calculate P and T axes for each focal mechanism.
        P, T = FocalMechanism(np.array([strike, dip, rake])).get_PT()
        Ps.append(P)
        Ts.append(T)

        if discrepancy_error:
            assert (
                df_stations is not None
            ), "df_stations must be provided for discrepancy error calculation."

    # Plotting conjugate nodal planes with the determined alpha values.
    for strike, dip, alph in zip(strikes_conj, dips_conj, alphas):
        ax.plane(strike, dip, auxiliary_line_style, linewidth=linewidth, alpha=alph)

    texts = [] # List to store text objects for station labels.

    # Plotting station polarities and amplitudes if station data is available.
    if df_stations is not None:
        # Set up colormap if color is to be used and 'amplitudes' values are present.
        if use_color and amp_values is not None:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) # Normalization for colormap.
            cmap = plt.get_cmap(palette) # Get the specified colormap.

        # Iterate through each station to plot its polarity and amplitude.
        for station, azimuth, takeoff, polarity, amp in zip(
            statns, azimuths, takeoffs, polarities, amp_values
        ):
            # Assign custom markers based on polarity value.
            if polarity == 0:
                marker = "o" # Circle for neutral polarity.
            elif polarity == 1:
                marker = "$+$" # Plus for positive polarity.
            else:  # polarity == -1
                marker = "$-$" # Minus for negative polarity.

            # Determine marker color based on 'use_color' and 'amplitudes' values.
            if use_color and amp_values is not None:
                color = cmap(norm(amp)) # Color from colormap based on 'amp' value.
            else:
                color = "black" # Default to black if no color mapping.

            # Plot the station as a line (representing the ray path) with a marker at the end.
            qq = ax.line(
                (90 - takeoff) % 180, # Convert takeoff angle to plunge for stereonet.
                azimuth,
                c=color,
                marker=marker,
                markersize=10,
            )
            # Get the coordinates of the plotted marker for text positioning.
            x_data, y_data = qq[0].get_xdata()[0], qq[0].get_ydata()[0]
            # Add station label text.
            text_obj = ax.text(
                x_data + 0.05, # Nudge text slightly from the marker.
                y_data + 0.05, # Nudge text slightly from the marker.
                station,
                fontsize=12,
                ha="center", # Horizontal alignment.
                va="center", # Vertical alignment.
            )
            texts.append(text_obj) # Add the text object to the list for adjustment.

        # Add a colorbar if color mapping was used for stations.
        if use_color and amp_values is not None:
            # Define colorbar axis position: [left, bottom, width, height].
            cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([]) # Dummy array for the scalar mappable.
            fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label="CP Value") # Label seems to be hardcoded as "CP Value", but it uses "amp_values" for coloring. This might be a typo.
            cbar_ax.yaxis.label.set_size(18) # Set font size for colorbar label.
            cbar_ax.set_ylim(0, vmax) # Set y-axis limits for the colorbar.

    # Plot P and T axes.
    if len(Ps) == 1: # If only one focal mechanism is provided.
        p_to = Ps[0][0] # Plunge of P-axis.
        p_az = Ps[0][1] # Azimuth of P-axis.
        t_to = Ts[0][0] # Plunge of T-axis.
        t_az = Ts[0][1] # Azimuth of T-axis.

        # Plot P-axis.
        ax.line(
            (90 - p_to) % 180, # Convert plunge to stereonet convention.
            p_az,
            c="black",
            marker=r"$P$", # P symbol as marker.
            markersize=12,
            alpha=alph, # Using the alpha from the single mechanism.
        )

        # Plot T-axis.
        ax.line(
            (90 - t_to) % 180,
            t_az,
            c="black",
            marker=r"$T$", # T symbol as marker.
            markersize=12,
            alpha=alph,
        )
    else: # If multiple focal mechanisms are provided.
        if jacknife:
            # Filter focal mechanisms based on 'kagan' value or 'Complete' index if jacknife is true.
            mask = (df_focalmechanism["kagan"] > 30) | (
                df_focalmechanism.index == "Complete"
            )
            mask = mask.values

            Ps = np.array(Ps)
            Ts = np.array(Ts)
            alphas = np.ones(len(Ps)) * 0.7 # Set a uniform alpha for jackknife P/T axes.
            Ps = Ps[mask] # Apply mask to P-axes.
            Ts = Ts[mask] # Apply mask to T-axes.
            alphas = alphas[mask] # Apply mask to alphas.

        # Plot each P and T axis.
        for i, (p, t, alph) in enumerate(zip(Ps, Ts, alphas)):
            p_to = p[0]
            p_az = p[1]
            t_to = t[0]
            t_az = t[1]

            # Plot P-axis with an index.
            ax.line(
                (90 - p_to) % 180,
                p_az,
                c="black",
                marker=rf"$P_{i}$", # P symbol with index.
                markersize=12,
                alpha=alph,
            )

            # Plot T-axis with an index.
            ax.line(
                (90 - t_to) % 180,
                t_az,
                c="black",
                marker=rf"$T_{i}$", # T symbol with index.
                markersize=12,
                alpha=alph,
            )

    # Adjust positions of text labels (station labels) to minimize overlaps.
    adjust_text_positions(ax, texts)

    ax.grid(True) # Display grid on the stereonet.

    # Set the plot title if provided.
    if title is not None:
        ax.set_title(title, fontsize=20) # Set title with a specific font size.

    return fig, ax


# %% [markdown]


def compute_conjugate_plane(strike: Union[float, np.ndarray], dip: Union[float, np.ndarray], rake: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes the conjugate plane parameters (strike, dip, rake) for a given 
    fault plane using standard numpy trigonometric functions.

    Args:
        strike (Union[float, np.ndarray]): The strike angle of the original plane in degrees.
        dip (Union[float, np.ndarray]): The dip angle of the original plane in degrees.
        rake (Union[float, np.ndarray]): The rake angle on the original plane in degrees.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the strike, dip, and rake angles of the conjugate plane, all in degrees.
    """
    # Convert input angles from degrees to radians for trigonometric calculations.
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)

    # Numerator for the arctan2 calculation of the new strike.
    top = np.cos(rake_rad) * np.sin(strike_rad) - np.cos(dip_rad) * np.sin(
        rake_rad
    ) * np.cos(strike_rad)
    # Denominator for the arctan2 calculation of the new strike.
    bot = np.cos(rake_rad) * np.cos(strike_rad) + np.cos(dip_rad) * np.sin(
        rake_rad
    ) * np.sin(strike_rad)

    # Calculate the new strike angle (strike_out) in degrees.
    strike_out = np.rad2deg(np.arctan2(top, bot))
    # Convert strike_out to a 'phi' angle (strike_out - 90) in radians, often used in formulas.
    phi = np.deg2rad(strike_out - 90)

    # Adjust strike_out if the original rake is negative, and normalize to [0, 360).
    strike_out = np.where(rake_rad < 0, strike_out - 180, strike_out)
    strike_out = strike_out % 360

    # Compute the new dip angle (dip_out) in degrees.
    dip_out = np.rad2deg(np.arccos(np.sin(np.abs(rake_rad)) * np.sin(dip_rad)))

    # Compute an intermediate value for the new rake angle.
    rake_out = -np.cos(phi) * np.sin(dip_rad) * np.sin(strike_rad) + np.sin(
        phi
    ) * np.sin(dip_rad) * np.cos(strike_rad)

    # Clip the intermediate rake_out value to ensure it's within valid arccos range [-1, 1].
    rake_out = np.clip(rake_out, -1.0, 1.0)

    # Calculate the final new rake angle in degrees, preserving the sign of the original rake.
    rake_out = np.rad2deg(np.copysign(np.arccos(rake_out), rake_rad))

    # Return the conjugate plane's strike (adjusted to [0, 360)), dip, and rake.
    return (strike_out - 90) % 360, dip_out, rake_out


def convert_azimuth_takeoffs_to_xy(azimuth: Union[float, np.ndarray], takeoff: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts azimuth and takeoff angles of a seismic ray to X, Y coordinates
    on an equal-area stereonet projection (Lambert azimuthal equal-area projection).

    Args:
        azimuth (Union[float, np.ndarray]): Azimuth angle(s) of the ray in degrees (0-360).
        takeoff (Union[float, np.ndarray]): Takeoff angle(s) of the ray from horizontal in degrees (0-90).

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
            A tuple containing the X and Y coordinates on the stereonet.
    """
    # Convert azimuth to radians (trend).
    trend_rad = np.deg2rad(azimuth)
    # Convert takeoff angle to plunge (90 - takeoff) and then to radians.
    plunge_rad = np.deg2rad((90 - takeoff) % 180)

    # Calculate colatitude (angular distance from the pole).
    colatitude = np.pi / 2 - plunge_rad

    # Calculate radius 'r' in the equal-area projection.
    r = np.sqrt(2.0) * np.sin(colatitude / 2)
    # Calculate X coordinate.
    x = r * np.sin(trend_rad)
    # Calculate Y coordinate.
    y = r * np.cos(trend_rad)

    return x, y


def proj_plane(strike: Union[float, np.ndarray], dip: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates X, Y coordinates for plotting a plane (great circle) on an
    equal-area stereonet projection given its strike and dip.

    Args:
        strike (Union[float, np.ndarray]): The strike angle(s) of the plane(s) in degrees.
        dip (Union[float, np.ndarray]): The dip angle(s) of the plane(s) in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing numpy arrays of X and Y coordinates that define the great circle(s).
    """
    # Convert strike and dip to radians.
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)

    # Generate an array of azimuths around the full circle for plotting the plane.
    azimuths = np.linspace(0, 360, num=361)
    azimuths_rad = np.deg2rad(azimuths)

    # Reshape strike_rad and dip_rad to be 2D arrays if they are 1D,
    # to enable broadcasting with azimuths_rad for vectorized computation.
    if strike_rad.ndim == 1:
        strike_rad = strike_rad[:, np.newaxis]
    if dip_rad.ndim == 1:
        dip_rad = dip_rad[:, np.newaxis]

    # Ensure azimuths_rad is correctly shaped for broadcasting if strike_rad is an array.
    if isinstance(strike_rad, np.ndarray):
        assert azimuths_rad.ndim == 1, "azimuths_rad must be a 1D array."
        assert strike_rad.ndim == 2, "strike_rad must be a 2D array."
        azimuths_rad = np.expand_dims(azimuths_rad, axis=0)
        azimuths_rad = np.repeat(azimuths_rad, strike_rad.shape[0], axis=0)

    # Calculate apparent dips for each azimuth around the plane.
    apparent_dips = np.arctan(np.tan(dip_rad) * np.sin(azimuths_rad - strike_rad))

    # Calculate colatitudes from apparent dips.
    colatitudes = np.pi / 2 - apparent_dips

    # Calculate radius 'r' in the equal-area projection.
    r = np.sqrt(2.0) * np.sin(colatitudes / 2)
    # Calculate X coordinates for the projected plane.
    x_proj = r * np.sin(azimuths_rad)
    # Calculate Y coordinates for the projected plane.
    y_proj = r * np.cos(azimuths_rad)

    return x_proj, y_proj


def is_left(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Determines if point p2 is to the left of the directed line from p0 to p1.

    Args:
        p0 (np.ndarray): Starting point(s) of the line segment(s). Shape (..., 2).
        p1 (np.ndarray): Ending point(s) of the line segment(s). Shape (..., 2).
        p2 (np.ndarray): Point(s) to check relative to the line segment(s). Shape (..., 2).

    Returns:
        np.ndarray: A scalar or array indicating the orientation:
                    > 0 if p2 is to the left of (p0, p1)
                    < 0 if p2 is to the right of (p0, p1)
                    = 0 if p2 is collinear with (p0, p1)
    """
    return (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1]) - (
        p2[..., 0] - p0[..., 0]
    ) * (p1[..., 1] - p0[..., 1])


def winding_number(polygon: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Calculates the winding number of a point with respect to a 2D polygon.
    This can be used to determine if a point is inside or outside a polygon
    (non-zero winding number implies inside for simple polygons).

    Args:
        polygon (np.ndarray): Vertices of the polygon. Expected shape (..., N, 2)
                              where N is the number of vertices and 2 for (x, y) coordinates.
        point (np.ndarray): The point(s) for which to calculate the winding number. Expected shape (..., 2).

    Returns:
        np.ndarray: The absolute winding number(s). 1 if inside, 0 if outside for simple polygons.
    """
    # Expand point dimensions to allow broadcasting with polygon segments.
    point = np.expand_dims(point, axis=-2)

    # Get current and next vertex of the polygon segments.
    p0 = polygon
    p1 = np.roll(polygon, shift=-1, axis=-2) # Shift by -1 to get the next point.

    # Determine upward and downward crossings of the horizontal line through the point.
    upward_crossing = (p0[..., 1] <= point[..., 1]) & (p1[..., 1] > point[..., 1])
    downward_crossing = (p0[..., 1] > point[..., 1]) & (p1[..., 1] <= point[..., 1])

    # Check orientation of the point relative to each polygon segment.
    isleft = is_left(p0, p1, point)

    # Calculate winding contributions for each segment.
    # +1 for upward crossing where point is to the left.
    # -1 for downward crossing where point is to the right.
    # 0 otherwise.
    winding_contributions = np.where(
        upward_crossing & (isleft > 0),
        1,
        np.where(downward_crossing & (isleft < 0), -1, 0),
    )

    # Sum contributions to get the total winding number.
    winding_number = np.sum(winding_contributions, axis=-1)

    # Return the absolute winding number (typically 1 for inside, 0 for outside).
    return np.abs(winding_number)


def compute_discrepancy_error(
    strikediprake: Union[Tuple, List, np.ndarray, torch.Tensor],
    polarity: np.ndarray,
    azimuths_stations: Union[np.ndarray, torch.Tensor],
    takeoffs_stations: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Computes the discrepancy error for a focal mechanism based on observed station
    polarities and their positions relative to the nodal planes.
    The error is the number of inconsistent polarities.

    Args:
        strikediprake (Union[Tuple, List, np.ndarray, torch.Tensor]):
            A tuple, list, numpy array, or torch tensor containing [strike, dip, rake] of the focal mechanism in degrees.
        polarity (np.ndarray): A numpy array of observed polarities at stations (1 for positive, -1 for negative).
        azimuths_stations (Union[np.ndarray, torch.Tensor]):
            A numpy array or torch tensor of station azimuths in degrees.
        takeoffs_stations (Union[np.ndarray, torch.Tensor]):
            A numpy array or torch tensor of station takeoff angles in degrees.

    Returns:
        np.ndarray: The total number of inconsistent polarities (discrepancy error).
    """
    # Convert strikediprake to a numpy array if it's a tuple, list, or torch.Tensor.
    if isinstance(strikediprake, tuple):
        strikediprake = np.array(strikediprake)
    elif isinstance(strikediprake, list):
        strikediprake = np.array(strikediprake)
    elif isinstance(strikediprake, torch.Tensor):
        strikediprake = strikediprake.cpu().numpy() # Move to CPU and convert to numpy.

    # Assert correct dimensions for strikediprake.
    assert (
        strikediprake.ndim == 1 and strikediprake.shape[0] == 3
    ), "strikediprake must be a 1D array with 3 elements."
    strike, dip, rake = strikediprake # Unpack focal mechanism parameters.

    # Convert station azimuths and takeoffs to numpy arrays if they are torch.Tensors.
    if isinstance(azimuths_stations, torch.Tensor):
        azimuths_stations = azimuths_stations.cpu().numpy()
    if isinstance(takeoffs_stations, torch.Tensor):
        takeoffs_stations = takeoffs_stations.cpu().numpy()
        
    # Ensure station azimuths and takeoffs are 1D arrays.
    if azimuths_stations.ndim > 1:
        azimuths_stations = np.squeeze(azimuths_stations)
        assert azimuths_stations.ndim == 1, "azimuths_stations must be a 1D array."
    if takeoffs_stations.ndim > 1:
        takeoffs_stations = np.squeeze(takeoffs_stations)
        assert takeoffs_stations.ndim == 1, "takeoffs_stations must be a 1D array."
        
    # Assert that station arrays have the same shape.
    assert (
        azimuths_stations.shape == takeoffs_stations.shape
    ), "azimuths_stations and takeoffs_stations must have the same shape."

    # Compute the conjugate plane.
    strike_conj, dip_conj, _ = compute_conjugate_plane(strike, dip, rake)

    # Convert station azimuths and takeoffs to stereonet X, Y coordinates.
    x_staz, y_staz = convert_azimuth_takeoffs_to_xy(
        azimuths_stations, takeoffs_stations
    )
    xy_staz = np.stack([x_staz, y_staz], axis=-1) # Stack to form (N, 2) array.

    # Project the primary nodal plane onto stereonet X, Y coordinates to form a polygon.
    x_plane1, y_plane1 = proj_plane(strike, dip)
    circ1 = np.expand_dims(np.stack([x_plane1, y_plane1], axis=-1), axis=-3)
    # Repeat the plane polygon for each station for vectorized winding number calculation.
    circ1_staz = np.repeat(circ1, xy_staz.shape[0], axis=-3)

    # Project the conjugate nodal plane onto stereonet X, Y coordinates to form a polygon.
    x_plane2, y_plane2 = proj_plane(strike_conj, dip_conj)
    circ2 = np.expand_dims(np.stack([x_plane2, y_plane2], axis=-1), axis=-3)
    # Repeat the conjugate plane polygon for each station.
    circ2_staz = np.repeat(circ2, xy_staz.shape[0], axis=-3)

    # Calculate winding numbers for stations with respect to both nodal planes.
    winding_numbers1 = winding_number(circ1_staz, xy_staz)
    winding_numbers2 = winding_number(circ2_staz, xy_staz)

    # Combine winding numbers to determine the region of the stereonet for each station.
    winding_numbers = (winding_numbers1 + winding_numbers2) % 2

    # Get the P (pressure) and T (tension) axes for the focal mechanism.
    P, T = FocalMechanism(strikediprake).get_PT()

    # Convert P and T axis plunge/azimuth to stereonet X, Y coordinates.
    xPT, yPT = convert_azimuth_takeoffs_to_xy(
        np.array([P[1], T[1]]), np.array([P[0], T[0]])
    )
    xy_PT = np.stack([xPT, yPT], axis=-1) # Stack to form (2, 2) array.

    # Repeat plane polygons for P and T axes for winding number calculation.
    circ1_PT = np.repeat(circ1, xy_PT.shape[0], axis=-3)
    circ2_PT = np.repeat(circ2, xy_PT.shape[0], axis=-3)
    # Calculate winding numbers for P and T axes with respect to both nodal planes.
    winding_numbers1_PT = winding_number(circ1_PT, xy_PT)
    winding_numbers2_PT = winding_number(circ2_PT, xy_PT)
    # Combine winding numbers for P and T axes.
    winding_numbers_PT = (winding_numbers1_PT + winding_numbers2_PT) % 2

    # Determine inconsistent polarities for P-wave arrivals.
    # A positive polarity (polarity == 1) is inconsistent if the station is in the dilation (T-axis) quadrant.
    areaP = np.where(
        np.logical_and(winding_numbers == winding_numbers_PT[0], polarity == 1), 1, 0
    )
    # Determine inconsistent polarities for T-wave arrivals.
    # A negative polarity (polarity == -1) is inconsistent if the station is in the compression (P-axis) quadrant.
    areaT = np.where(
        np.logical_and(winding_numbers == winding_numbers_PT[1], polarity == -1), 1, 0
    )
    
    # Sum the inconsistent polarities to get the total discrepancy error.
    return np.sum(areaP) + np.sum(areaT)


# %%


def compute_conjugate_plane_torch(strike: torch.Tensor, dip: torch.Tensor, rake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the conjugate plane parameters (strike, dip, rake) for a given
    fault plane using PyTorch tensor operations.

    Args:
        strike (torch.Tensor): The strike angle of the original plane in degrees.
        dip (torch.Tensor): The dip angle of the original plane in degrees.
        rake (torch.Tensor): The rake angle on the original plane in degrees.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the strike, dip, and rake angles of the conjugate plane, all in degrees.
    """
    # Convert input angles from degrees to radians.
    strike_rad = torch.deg2rad(strike)
    dip_rad = torch.deg2rad(dip)
    rake_rad = torch.deg2rad(rake)

    # Slip Vector Computation (Numerator and Denominator for arctan2)
    top = torch.cos(rake_rad) * torch.sin(strike_rad) - torch.cos(dip_rad) * torch.sin(
        rake_rad
    ) * torch.cos(strike_rad)
    bot = torch.cos(rake_rad) * torch.cos(strike_rad) + torch.cos(dip_rad) * torch.sin(
        rake_rad
    ) * torch.sin(strike_rad)

    # Calculate the new strike angle (strike_out) in degrees.
    strike_out = torch.rad2deg(torch.atan2(top, bot))
    # Convert strike_out to a 'phi' angle (strike_out - 90) in radians.
    phi = torch.deg2rad(strike_out - 90)

    # Adjust strike_out if rake is negative, and normalize to [0, 360).
    strike_out = torch.where(rake_rad < 0, strike_out - 180, strike_out)
    strike_out = strike_out % 360

    # Compute the new dip angle (dip_out) in degrees.
    dip_out = torch.rad2deg(
        torch.acos(torch.sin(torch.abs(rake_rad)) * torch.sin(dip_rad))
    )

    # Compute an intermediate value for the new rake angle.
    rake_out = -torch.cos(phi) * torch.sin(dip_rad) * torch.sin(strike_rad) + torch.sin(
        phi
    ) * torch.sin(dip_rad) * torch.cos(strike_rad)

    # Clamp the intermediate rake_out value to ensure it's within valid arccos range [-1, 1].
    rake_out = torch.clamp(rake_out, min=-1.0, max=1.0)

    # Calculate the final new rake angle in degrees, preserving the sign of the original rake.
    rake_out = torch.rad2deg(torch.copysign(torch.acos(rake_out), rake_rad))

    # Return the conjugate plane's strike (adjusted to [0, 360)), dip, and rake.
    return (strike_out - 90) % 360, dip_out, rake_out


def convert_azimuth_takeoffs_to_xy_torch(azimuth: torch.Tensor, takeoff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts azimuth and takeoff angles of a seismic ray to X, Y coordinates
    on an equal-area stereonet projection using PyTorch tensor operations.

    Args:
        azimuth (torch.Tensor): Azimuth angle(s) of the ray in degrees (0-360).
        takeoff (torch.Tensor): Takeoff angle(s) of the ray from horizontal in degrees (0-90).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A tuple containing the X and Y coordinates on the stereonet.
    """
    # Convert azimuth to radians (trend).
    trend_rad = torch.deg2rad(azimuth)
    # Convert takeoff angle to plunge (90 - takeoff) and then to radians.
    plunge_rad = torch.deg2rad((90 - takeoff) % 180)

    # Calculate colatitude (angular distance from the pole).
    colatitude = torch.pi / 2 - plunge_rad

    # Calculate radius 'r' in the equal-area projection.
    r = torch.sqrt(torch.tensor(2.0)) * torch.sin(colatitude / 2)
    # Calculate X coordinate.
    x = r * torch.sin(trend_rad)
    # Calculate Y coordinate.
    y = r * torch.cos(trend_rad)

    return x, y


def proj_plane_torch(strike: torch.Tensor, dip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates X, Y coordinates for plotting a plane (great circle) on an
    equal-area stereonet projection given its strike and dip, using PyTorch.

    Args:
        strike (torch.Tensor): The strike angle(s) of the plane(s) in degrees.
        dip (torch.Tensor): The dip angle(s) of the plane(s) in degrees.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A tuple containing torch tensors of X and Y coordinates that define the great circle(s).
    """
    # Convert the strike and dip to radians.
    strike_rad = torch.deg2rad(strike)
    dip_rad = torch.deg2rad(dip)

    # Generate an array of azimuths around the full circle for plotting the plane.
    azimuths = torch.linspace(0, 360, steps=361)
    azimuths_rad = torch.deg2rad(azimuths)
    
    # Expand dimensions and repeat azimuths_rad to match the batch dimension of strike_rad.
    azimuths_rad = azimuths_rad.unsqueeze(0).repeat(strike_rad.shape[0], 1)

    # Calculate apparent dips for each azimuth around the plane.
    apparent_dips = torch.atan(
        torch.tan(dip_rad) * torch.sin(azimuths_rad - strike_rad)
    )

    # Calculate colatitudes from apparent dips.
    colatitudes = torch.pi / 2 - apparent_dips

    # Calculate radius 'r' in the equal-area projection.
    r = torch.sqrt(torch.tensor(2.0)) * torch.sin(colatitudes / 2)
    # Calculate X coordinates for the projected plane.
    x_proj = r * torch.sin(azimuths_rad)
    # Calculate Y coordinates for the projected plane.
    y_proj = r * torch.cos(azimuths_rad)

    return x_proj, y_proj


def is_left_torch(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Determines if point p2 is to the left of the directed line from p0 to p1 using PyTorch.

    Args:
        p0 (torch.Tensor): Starting point(s) of the line segment(s). Shape (..., 2).
        p1 (torch.Tensor): Ending point(s) of the line segment(s). Shape (..., 2).
        p2 (torch.Tensor): Point(s) to check relative to the line segment(s). Shape (..., 2).

    Returns:
        torch.Tensor: A tensor indicating the orientation:
                      > 0 if p2 is to the left of (p0, p1)
                      < 0 if p2 is to the right of (p0, p1)
                      = 0 if p2 is collinear with (p0, p1)
    """
    return (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1]) - (
        p2[..., 0] - p0[..., 0]
    ) * (p1[..., 1] - p0[..., 1])


def winding_number_torch(polygon: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Calculates the winding number of a point with respect to a 2D polygon using PyTorch.
    This can be used to determine if a point is inside or outside a polygon.

    Args:
        polygon (torch.Tensor): Vertices of the polygon. Expected shape (..., N, 2)
                                where N is the number of vertices and 2 for (x, y) coordinates.
        point (torch.Tensor): The point(s) for which to calculate the winding number. Expected shape (..., 2).

    Returns:
        torch.Tensor: The absolute winding number(s). 1 if inside, 0 if outside for simple polygons.
    """
    # Expand point dimensions to allow broadcasting with polygon segments.
    point = point.unsqueeze(2)

    # Get current and next vertex of the polygon segments.
    p0 = polygon
    p1 = torch.roll(polygon, shifts=-1, dims=2) # Shift by -1 to get the next point along dimension 2.

    # Determine upward and downward crossings of the horizontal line through the point.
    upward_crossing = (p0[..., 1] <= point[..., 1]) & (p1[..., 1] > point[..., 1])
    downward_crossing = (p0[..., 1] > point[..., 1]) & (p1[..., 1] <= point[..., 1])

    # Check orientation of the point relative to each polygon segment.
    isleft = is_left_torch(p0, p1, point)

    # Calculate winding contributions for each segment using torch.where.
    winding_contributions = torch.where(
        upward_crossing & (isleft > 0),
        1,
        torch.where(downward_crossing & (isleft < 0), -1, 0),
    )

    # Sum contributions to get the total winding number along dimension 2.
    winding_number = torch.sum(winding_contributions, dim=2)
    return winding_number.abs() # Return the absolute winding number.


def compute_discrepancy_error_torch(
    strikediprake: Union[Tuple, List, np.ndarray, torch.Tensor],
    polarity: torch.Tensor,
    azimuths_stations: torch.Tensor,
    takeoffs_stations: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the discrepancy error for a focal mechanism based on observed station
    polarities and their positions relative to the nodal planes, using PyTorch.
    The error is the number of inconsistent polarities.

    Args:
        strikediprake (Union[Tuple, List, np.ndarray, torch.Tensor]):
            A tuple, list, numpy array, or torch tensor containing [strike, dip, rake] of the focal mechanism in degrees.
        polarity (torch.Tensor): A torch tensor of observed polarities at stations (1 for positive, -1 for negative).
        azimuths_stations (torch.Tensor):
            A torch tensor of station azimuths in degrees.
        takeoffs_stations (torch.Tensor):
            A torch tensor of station takeoff angles in degrees.

    Returns:
        torch.Tensor: The total number of inconsistent polarities (discrepancy error) as a tensor.
    """
    # Convert strikediprake to a torch.Tensor if it's not already.
    if isinstance(strikediprake, tuple):
        strikediprake = torch.tensor(strikediprake)
    elif isinstance(strikediprake, list):
        strikediprake = torch.tensor(strikediprake)
    elif isinstance(strikediprake, np.ndarray):
        strikediprake = torch.tensor(strikediprake)

    # Compute the conjugate plane using PyTorch.
    # Assuming strikediprake is (batch_size, 3) or (3,)
    strike_conj, dip_conj, _ = compute_conjugate_plane_torch(
        *strikediprake.split(1, dim=1)
    )
    # Extract strike, dip from original strikediprake tensor.
    strike, dip, _ = strikediprake.split(1, dim=1)

    # Convert station azimuths and takeoffs to stereonet X, Y coordinates using PyTorch.
    x_staz, y_staz = convert_azimuth_takeoffs_to_xy_torch(
        azimuths_stations, takeoffs_stations
    )
    xy_staz = torch.stack([x_staz, y_staz], dim=-1) # Stack to form (N, 2) or (batch, N, 2) array.

    # Project the primary nodal plane onto stereonet X, Y coordinates to form a polygon.
    x_plane1, y_plane1 = proj_plane_torch(strike, dip)
    circ1 = torch.stack([x_plane1, y_plane1], dim=-1)
    # Repeat the plane polygon for each station for vectorized winding number calculation.
    circ1_staz = circ1.unsqueeze(1).repeat(1, xy_staz.shape[1], 1, 1) # Assumes circ1 is (batch, N_points, 2)

    # Project the conjugate nodal plane onto stereonet X, Y coordinates to form a polygon.
    x_plane2, y_plane2 = proj_plane_torch(strike_conj, dip_conj)
    circ2 = torch.stack([x_plane2, y_plane2], dim=-1)
    # Repeat the conjugate plane polygon for each station.
    circ2_staz = circ2.unsqueeze(1).repeat(1, xy_staz.shape[1], 1, 1)

    # Calculate winding numbers for stations with respect to both nodal planes.
    winding_numbers1 = winding_number_torch(circ1_staz, xy_staz)
    winding_numbers2 = winding_number_torch(circ2_staz, xy_staz)

    # Combine winding numbers to determine the region of the stereonet for each station.
    winding_numbers = (winding_numbers1 + winding_numbers2) % 2

    # Get the P (pressure) and T (tension) axes for the focal mechanism using PyTorch.
    P, T = FocalMechanism_torch(strikediprake).get_PT()

    # Convert P and T axis plunge/azimuth to stereonet X, Y coordinates.
    # Note: P and T from FocalMechanism_torch are (batch_size, 2) [plunge, azimuth]
    xPT, yPT = convert_azimuth_takeoffs_to_xy_torch(
        torch.stack([P[:,1], T[:,1]], dim=-1), # Stack azimuths
        torch.stack([P[:,0], T[:,0]], dim=-1) # Stack plunges (takeoffs in stereonet convention)
    )
    xy_PT = torch.stack([xPT, yPT], dim=-1) # Stack to form (batch_size, 2, 2)

    # Compute the winding number for the P and T points.
    circ1_PT = circ1.unsqueeze(1).repeat(1, xy_PT.shape[1], 1, 1)
    circ2_PT = circ2.unsqueeze(1).repeat(1, xy_PT.shape[1], 1, 1)
    winding_numbers1_PT = winding_number_torch(circ1_PT, xy_PT)
    winding_numbers2_PT = winding_number_torch(circ2_PT, xy_PT)
    winding_numbers_PT = (winding_numbers1_PT + winding_numbers2_PT) % 2 # Shape (batch_size, 2)

    # Compute the discrepancy error by checking consistency of station polarities.
    # Check consistency for positive polarities (P-wave).
    # Assumes winding_numbers_PT[0, 0] corresponds to the P-axis region
    # and winding_numbers_PT[0, 1] corresponds to the T-axis region.
    # This logic implies 'winding_numbers' is 0 for dilation and 1 for compression or vice-versa.
    areaP = torch.where(
        torch.logical_and(winding_numbers == winding_numbers_PT[0, 0], polarity == 1), 1, 0
    )
    # Check consistency for negative polarities (S-wave related).
    areaT = torch.where(
        torch.logical_and(winding_numbers == winding_numbers_PT[0, 1], polarity == -1), 1, 0
    )

    # Sum the inconsistent polarities to get the total discrepancy error for each focal mechanism in the batch.
    return areaP.sum(-1) + areaT.sum(-1)


# %%
