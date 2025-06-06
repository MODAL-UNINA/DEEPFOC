#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
from pathlib import Path

# %%


def read_velmod_data(
    filepath: Path, dtype: type = np.float32
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Read velocity model data from a text file.

    The file is expected to start with a header line containing:
        - A latitude reference (as a string)
        - A longitude reference (as a string)
        - The number of subsequent data lines (nl)
    Each of the following nl lines contains two values:
        - A velocity value (vl)
        - A depth value (dl)
    These values are converted to the specified dtype.

    Args:
        filepath (Path): Path to the velocity model data file.
        dtype (type, optional): Numeric type for conversion (e.g., np.float32).
            Default is np.float32.

    Returns:
        tuple:
            - latref (float): Latitude reference value.
            - lonref (float): Longitude reference value.
            - v (np.ndarray): Array of velocity values.
            - d (np.ndarray): Array of depth values.
    """
    # Initialize lists to collect velocity and depth values.
    v_l = []
    d_l = []

    # Open the file and read the header and data lines.
    with open(filepath, "r") as file:
        # Read the first line which contains the header information.
        latref_s, lonref_s, nl_s = next(file).split()
        # Convert the latitude and longitude references to the desired type.
        latref, lonref = dtype.type(latref_s), dtype.type(lonref_s)
        # The number of lines containing data.
        nl = int(nl_s)
        # Loop over the next nl lines and convert each pair of values.
        for _ in range(nl):
            # Convert both velocity and depth values to the specified dtype.
            vl, dl = [dtype.type(x) for x in next(file).split()]
            v_l.append(vl)
            d_l.append(dl)

    # Convert the lists to numpy arrays.
    v = np.array(v_l)
    d = np.array(d_l)
    return latref, lonref, v, d


def read_stations_data(
    filepath: Path, dtype: type = np.float32
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Read station information data from a text file.

    Each line in the file is expected to have exactly 4 entries:
        - Station name (string)
        - Longitude (as a string, converted to numeric)
        - Latitude (as a string, converted to numeric)
        - Elevation (as a string, converted to numeric)
    The numeric values are converted to the specified dtype.

    Args:
        filepath (Path): Path to the stations data file.
        dtype (type, optional): Numeric type for conversion (e.g., np.float32).
            Default is np.float32.

    Returns:
        tuple:
            - staz (list[str]): List of station names.
            - lats (np.ndarray): Array of station latitudes.
            - lons (np.ndarray): Array of station longitudes.
            - elev (np.ndarray): Array of station elevations.
    """
    # Initialize lists to collect station names and numerical data.
    staz_l = []
    lats_l = []
    lons_l = []
    elev_l = []

    # Open the file and process each line.
    with open(filepath, "r") as file:
        for line in file:
            parts = line.split()
            # Stop processing if the line doesn't have exactly 4 parts.
            if len(parts) != 4:
                break
            # Append the station name.
            staz_l.append(parts[0])
            # Convert and append longitude, latitude, and elevation.
            lons_l.append(dtype.type(parts[1]))
            lats_l.append(dtype.type(parts[2]))
            elev_l.append(dtype.type(parts[3]))

    # Convert the numerical lists to numpy arrays.
    lats = np.array(lats_l)
    lons = np.array(lons_l)
    elev = np.array(elev_l)

    # Return the station names list and the corresponding arrays.
    return staz_l, lats, lons, elev


# %%
