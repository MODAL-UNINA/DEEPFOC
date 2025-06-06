#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% [Imports]
import os
import shutil
import threading
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures
import scipy.special as sc
from tqdm.auto import tqdm
from IPython.display import display

from typing import Union, Optional, List, Dict, Any, Tuple

# %% [Constants and Global Variables]
# Earth radius in kilometers and flattening parameter (ellipticity)
R = 6378.388
E = 0.0033670033
# Conversion factor from degrees to radians
PO180 = np.pi / 180

# Set the library directory to the directory where this file is located
library_directory = Path(__file__).resolve().parent


def get_library_directory() -> Path:
    """
    Returns the directory in which the library (this module) resides.
    """
    return library_directory


def to_deg_min(coord: float) -> str:
    """
    Converts a coordinate in decimal degrees to a string format "degrees-minutes".

    Parameters:
        coord (float): The coordinate in decimal degrees.

    Returns:
        str: The coordinate formatted as "degrees-minutes" (e.g., "47-25.26").
    """
    degrees = int(coord)  # Extract integer degree portion.
    minutes = round((coord - degrees) * 60, 2)  # Convert fractional degree to minutes.
    return f"{degrees}-{minutes:05.2f}"  # Format as "degrees-minutes".


def read_velmod_data(
    filepath: Path, dtype=np.float32
) -> Tuple[np.floating, np.floating, np.ndarray, np.ndarray]:
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
        Tuple:
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
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple:
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


def fr(tdeg: np.floating):
    """
    Function to calculate the value of fr based on input t.
    """
    float_type = tdeg.dtype.type
    return float_type(1.0) - float_type(E) * sc.sindg(tdeg) ** float_type(2.0)


def check_and_gclc(
    zlatdeg: np.floating, zlondeg: np.floating, xlatdeg: np.ndarray, ylondeg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts geographical coordinates.
    """
    if not isinstance(zlatdeg, np.floating):
        raise ValueError("zlatdeg must be a float")
    if not isinstance(zlondeg, np.floating):
        raise ValueError("zlondeg must be a float")
    if not isinstance(xlatdeg, np.ndarray):
        raise ValueError("xlatdeg must be an array")
    if not isinstance(ylondeg, np.ndarray):
        raise ValueError("ylondeg must be an array")

    if xlatdeg.ndim != 1:
        raise ValueError("xlatdeg must be a 1D array")
    if ylondeg.ndim != 1:
        raise ValueError("ylondeg must be a 1D array")

    if len(set(map(len, (xlatdeg, ylondeg)))) != 1:
        raise ValueError("xlatdeg and ylondeg must have the same length")
    return gclc(zlatdeg, zlondeg, xlatdeg, ylondeg)


def gclc(
    zlatdeg: np.floating, zlondeg: np.floating, xlatdeg: np.ndarray, ylondeg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts geographical coordinates.
    """
    dtypes = [zlatdeg.dtype, zlondeg.dtype, xlatdeg.dtype, ylondeg.dtype]

    if any(dt.name.startswith(("uint", "int")) for dt in dtypes):
        raise ValueError("radpar: Input types must be floating point")

    float_dtype = xlatdeg.dtype
    if len(set(dtypes)) != 1:
        print("Input types not matching. Promoting to the highest precision")
        float_dtype = max(dtypes, key=lambda x: x.itemsize)
    float_type = float_dtype.type

    po180 = float_type(PO180)
    r = float_type(R)
    e = float_type(E)

    # Convert input degrees to radians
    zlat = zlatdeg * po180
    zlon = zlondeg * po180
    xlat = xlatdeg * po180
    ylon = ylondeg * po180
    # Calculations
    crlatdeg = np.arctan(e * sc.sindg(float_type(2.0) * zlatdeg) / fr(zlatdeg)) / po180
    zgcldeg = zlatdeg - crlatdeg
    a = fr(zgcldeg)
    rho = r * a
    b = (e * sc.sindg(float_type(2.0) * zgcldeg) / a) ** float_type(2.0) + float_type(
        1.0
    )
    c = float_type(2.0) * e * sc.cosdg(float_type(2.0) * zgcldeg) * a + (
        e * sc.sindg(float_type(2.0) * zgcldeg)
    ) ** float_type(2.0)
    cdist = c / (a ** float_type(2.0) * b) + float_type(1.0)

    # NOTE some values of cos(xlat - crlat) diverge from Fortran
    # output (in float32) by 1 in the integer representation.
    cos_factor = (
        sc.cosdg((xlatdeg - crlatdeg).astype(np.float64)).astype(float_dtype)
        if float_dtype.itemsize == 4  # float32
        else sc.cosdg(xlatdeg - crlatdeg)
    )

    yp = -rho * (zlon - ylon) * cos_factor
    xp = rho * (xlat - zlat) / cdist

    return xp, yp


def inverse_gclc(
    xp: np.ndarray, 
    yp: np.ndarray, 
    zlatdeg: float, 
    zlondeg: float, 
    max_iter: int = 10, 
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inversely converts projected coordinates (xp, yp) back to geographical coordinates (latitude, longitude).
    Attention: The code correctly convert the latitude but there are some differences in the longitude values.

    Parameters:
        xp, yp: Projected coordinate arrays.
        zlatdeg, zlondeg: Reference latitude and longitude in degrees.
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Tolerance for convergence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (xlatdeg, ylondeg) where both are arrays of geographical
            coordinates in degrees.
    """
    float_type = np.float64  # Use double precision for the inverse calculation

    po180 = float_type(PO180)
    r = float_type(R)
    e = float_type(E)

    # Convert reference coordinates from degrees to radians
    zlat = zlatdeg * po180
    zlon = zlondeg * po180

    # Compute correction for latitude as in the forward conversion
    crlatdeg = np.arctan(e * np.sin(2.0 * zlatdeg * po180) / fr(zlatdeg)) / po180
    zgcldeg = zlatdeg - crlatdeg

    # Compute scaling factors (rho and cdist)
    a = fr(zgcldeg)
    rho = r * a
    b = (e * np.sin(2.0 * zgcldeg * po180) / a) ** 2 + 1.0
    c = (
        2.0 * e * np.cos(2.0 * zgcldeg * po180) * a
        + (e * np.sin(2.0 * zgcldeg * po180)) ** 2
    )
    cdist = c / (a**2 * b) + 1.0

    # Inverse calculation for latitude (xlat)
    xlat = (xp * cdist / rho) + zlat

    # Iterative inverse calculation for longitude (ylon)
    ylon = zlon
    for _ in range(max_iter):
        cos_term = np.cos(xlat - crlatdeg)
        ylon_new = zlon - (yp / (-rho * cos_term))
        if np.allclose(ylon_new, ylon, atol=tol):
            break
        ylon = ylon_new

    # Convert results from radians back to degrees
    xlatdeg = xlat / po180
    ylondeg = ylon / po180

    return xlatdeg, ylondeg


# %% [FPfit Execution Functions]

def run_fpfit(
    filename: Union[str, Path] = "tempfpfit.prt",
    out_filename: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Runs the external FPfit executable along with subsequent processing steps.

    This function performs the following steps:
        1. Prepares an input file (.inp) required by FPfit.
        2. Changes directory to the location of the input file.
        3. Executes FPfit via a subprocess call.
        4. Runs FPplot to generate a PostScript (.ps) file from the FPfit results.
        5. Converts the PostScript file to a PDF.
        6. Reads the resulting .fps file to extract strike, dip, and rake values.

    Parameters:
        filename (str or Path): The name or path of the file containing event data.
        out_filename (str or Path): The base name for output files produced by FPfit.
        verbose (bool): If True, print diagnostic messages during execution.

    Returns:
        list: A list of dictionaries, where each dictionary contains the extracted solution
              parameters (strike, dip, rake, slip) from the FPfit output.

    Raises:
        RuntimeError: If any subprocess call (fpfit, fpplot, or ps2pdf) fails.
        FileNotFoundError: If any required output file is not generated.
    """
    # Convert 'filename' to a Path object and obtain its parent directory.
    if isinstance(filename, str):
        filename_path = Path().resolve() / filename
    elif isinstance(filename, Path):
        filename_path = filename
        filename = filename_path.name

    # Determine the output filename and, if provided, the output directory.
    if isinstance(out_filename, Path):
        out_filename_path = out_filename.parent
        out_filename = out_filename.name
    elif isinstance(out_filename, str):
        out_filename_path = None
    else:
        # If out_filename is not provided, derive it from the input filename.
        out_filename = "".join(filename.split(".")[:-1])
        out_filename_path = None

    try:
        # ---------------------------------------------------------------------
        # Step 1: Prepare the input file (.inp) for FPfit.
        # ---------------------------------------------------------------------
        # The .inp file tells FPfit which files to use for input and where to write its outputs.
        inp_filename = f"{filename}.inp"
        # Open the new .inp file for writing in the same directory as the input file.
        with open(filename_path.parent / inp_filename, "w") as inp_file:
            # Write required FPfit commands and file specifications.
            inp_file.write("\n")
            inp_file.write("ttl   1 'none'\n")
            inp_file.write(f'hyp "{filename}"\n')
            inp_file.write(f'out "{out_filename}.out"\n')
            inp_file.write(f'pol "{out_filename}.pol"\n')
            inp_file.write(f'sum "{out_filename}.fps"\n')
            # Append additional commands from a pre-existing template file.
            with open(filename_path.parent / "run_fpfit_input.inp", "r") as extra_input:
                inp_file.write(extra_input.read())

        # if verbose:
            # print("Running fpfit...")

        # Change the working directory to where the input file is located.
        os.chdir(filename_path.parent)

        # ---------------------------------------------------------------------
        # Step 2: Run FPfit using the prepared input file.
        # ---------------------------------------------------------------------
        # The input string instructs FPfit to read the .inp file and then execute 'fps' and 'sto' commands.
        result = subprocess.run(
            ["./fpfit"],
            input=f"@{inp_filename}\nfps\nsto\n",
            text=True,
            capture_output=True,
        )
        # If FPfit returns a non-zero exit status, raise an error.
        if result.returncode != 0:
            raise RuntimeError(f"Error running fpfit: {result.stderr.decode()}")
        elif verbose:
            print(result.stdout)

        # Verify that the .pol file (which should contain polarity data) was generated and is not empty.
        if (
            not os.path.exists(f"{out_filename}.pol")
            or os.path.getsize(f"{out_filename}.pol") == 0
        ):
            raise FileNotFoundError(f"Output file {out_filename}.pol not generated")

        if verbose:
            print("Done. Running fpplot...")
        
        # ---------------------------------------------------------------------
        # Step 3: Run FPplot to convert the FPfit output into a PostScript (.ps) file.
        # ---------------------------------------------------------------------
        os_env = os.environ.copy()
        os_env["POSTSCRIPT_FILE_NAME"] = f"{out_filename}.ps"
        result = subprocess.run(
            ["./fpplot"],
            input=f"{out_filename}.pol\ny\n\n\n\n\n2\nstop\n",
            text=True,
            capture_output=True,
            env=os_env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error running fpplot: {result.stderr}")

        if verbose:
            print("Done. Converting the PostScript file to PDF...")

        # ---------------------------------------------------------------------
        # Step 4: Convert the PostScript (.ps) file to a PDF using 'ps2pdf'.
        # ---------------------------------------------------------------------
        result = subprocess.run(["ps2pdf", f"{out_filename}.ps"], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error converting PostScript to PDF: {result.stderr.decode()}"
            )

        if verbose:
            print(f"Done. File: {out_filename}.pdf")

        # ---------------------------------------------------------------------
        # Step 5: Read and parse the .fps file to extract focal mechanism parameters.
        # ---------------------------------------------------------------------
        fps_filename = f"{out_filename}.fps"
        if not os.path.exists(fps_filename):
            raise FileNotFoundError(f"Output file {fps_filename} not generated")

        if verbose:
            print("Reading .fps file content for debugging...")
        with open(fps_filename, "r") as fps_file:
            fps_content = fps_file.readlines()

        if verbose:
            print("Content of .fps file:")
        # Optionally print each line if verbose mode is enabled.
        for line in fps_content:
            if verbose:
                print(line.strip())

        # Initialize an empty list to collect solution dictionaries.
        solutions = []
        # Loop over each line of the .fps file.
        for line in fps_content:
            try:
                parts = line.split()
                # Only process lines that have more than 15 parts (ensuring enough columns).
                if len(parts) > 15:  # Ensure the line has enough columns
                    # If the last element is not "*", append "*" to standardize the format.
                    if parts[-1] != "*":
                        parts.append("*")
                    # Correct formatting if the expected column has extra characters.
                    if len(parts[-10]) > 4:
                        temp = parts[-10].split("-")
                        parts[-10] = "-" + temp[1]
                        parts.insert(-10, temp[0])

                    # Parse the slip value from the corresponding column (index may require adjustment).
                    slip = float(parts[-12])  # Adjust index based on actual format
                    # Compute the strike by subtracting 90 and wrapping modulo 360.
                    strike = (slip - 90) % 360
                    # Parse dip and rake values from their respective positions.
                    dip = float(parts[-11])  # Adjust index based on actual format
                    rake = float(parts[-10])  # Adjust index based on actual format
                    # Append the extracted solution as a dictionary.
                    solutions.append(
                        {"strike": strike, "dip": dip, "rake": rake, "slip": slip}
                    )
            except (IndexError, ValueError) as parse_error:
                print(f"Parsing error: {parse_error}")

        # If no valid solutions were extracted, raise an error.
        if not solutions:
            raise ValueError(
                "Could not extract any strike, dip, and rake values from the .fps file"
            )

        if verbose:
            display("Results:", solutions)
        # If an output directory was specified, copy the generated PDF there.
        if out_filename_path:
            shutil.copy(
                f"{out_filename}.pdf", out_filename_path / f"{out_filename}.pdf"
            )
            if verbose:
                print(f"File copied to {out_filename_path}")

        # Return the list of extracted solutions.
        return solutions

    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception(f"An error occurred: {e}")


def fpfit(
    event: pd.Series,  # A pandas Series representing a seismic event; must have attributes like lat, lon, x, y, z and keys for polarities, azimuth, etc.
    df_stations: pd.DataFrame,  # DataFrame with station data (columns like staz, xstaz, ystaz, etc.)
    mag_str: str,  # Magnitude string (e.g., "3.5")
    pol_cols: List[
        str
    ],  # List of column names for polarities (e.g., ["pol_ST01", "pol_ST02", ...])
    latref_lonref: Optional[
        Union[List[float], tuple]
    ] = None,  # Optional reference latitude and longitude as a 2-element list or tuple
    file_name: Union[
        str, Path
    ] = "tempfpfit.prt",  # Base filename for temporary FPfit input files
) -> List[Dict[str, Any]]:
    """
    Prepares the input file for FPfit and runs the FPfit inversion for a single seismic event.

    This function formats the event data into a specific text structure expected by FPfit,
    writes it to a file, calls the external FPfit executable, and returns the extracted solutions.

    Parameters:
        event (pd.Series): A row representing the event, with attributes such as lat, lon, x, y, z,
                           and keys like f"pol_{station}" for station polarities.
        df_stations (pd.DataFrame): DataFrame containing station data with columns including 'staz', 'xstaz', and 'ystaz'.
        mag_str (str): Magnitude as a string.
        pol_cols (List[str]): List of polarity column names.
        latref_lonref (Optional[Union[List[float], tuple]]): Optional reference latitude and longitude for coordinate conversion.
        file_name (Union[str, Path]): The base filename for the temporary FPfit input file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the extracted solution parameters
                              ('strike', 'dip', 'rake', 'slip') from the FPfit inversion.
    """
    # If file_name is a string, build a folder path based on its name (excluding the extension)
    if isinstance(file_name, str):
        folder_path = (library_directory / "".join(file_name.split(".")[:-1])).resolve()
        file_name = folder_path / file_name

    # Initialize the header string for FPfit input.
    out = "  DATE    ORIGIN    LAT N    LONG E    DEPTH    MAG NO STRIKE DIP RAKE STAZFLIP  \n"

    # Determine the latitude and longitude strings.
    if latref_lonref is None:
        # Use the event's lat and lon attributes and convert them to a "degrees-minutes" string.
        lat_s = to_deg_min(event.lat)
        lon_s = to_deg_min(event.lon)
    elif len(latref_lonref) == 2:
        # FIXME: The conversion here is known to produce discrepancies between the original
        # longitude and the converted values. But for the purposes of using the algorithm,
        # it does not affect the results
        lat_s, lon_s = inverse_gclc(
            event.x, event.y, latref_lonref[0], latref_lonref[1]
        )
        lat_s = to_deg_min(lat_s)
        lon_s = to_deg_min(lon_s)

    elev_s = str(round(event.z, 2))
    # Adjust spacing based on the length of the degree portion.
    lon_space = "   " if len(lon_s.split("-")[0]) == 1 else "  "
    elev_space = "   " if len(elev_s.split(".")[0]) == 1 else "  "
    # Append event header line with formatted coordinates and magnitude information.
    out += f" 240830  917 36.42 {lat_s}{lon_space}{lon_s}{elev_space}{elev_s} {mag_str} {(event[pol_cols] != 0).sum()}\n"
    out += "\n"
    # Append station header.
    out += "  STN  DIST AZM AIN PRMK"

    # Loop over each station in df_stations and append station-specific data.
    for i, stat in df_stations.iterrows():
        # Retrieve the polarity for the current station from the event.
        pol = event[f"pol_{stat.staz}"]

        # Skip stations with zero polarity.
        if pol == 0:
            continue
        # Get azimuth and takeoff angle for the station.
        az = event[f"az_{stat.staz}"]
        ih = event[f"ih_{stat.staz}"]
        # Determine polarity label: "U" for positive, "D" for negative, or blank.
        ppol = (
            "U"
            if event[f"pol_{stat.staz}"] > 0
            else "D" if event[f"pol_{stat.staz}"] < 0 else " "
        )
        # Compute the distance between the station and the event using Euclidean distance.
        dist = np.sqrt((stat.xstaz - event.x) ** 2 + (stat.ystaz - event.y) ** 2).item()
        # Append formatted station data to the output string.
        out += f"\n {stat.staz} {dist:>5.1f} {int(np.round(az)):>3} {int(np.round(ih)):>3}  P{ppol}0 "

    out += "\n\n"

    # Write the complete FPfit input text to the file.
    with open(file_name, "w") as f:
        f.write(out)

    # Execute FPfit using the prepared input file and return its parsed results.
    result = run_fpfit(file_name)

    return result


def fpyfit(
    df_real_test: pd.DataFrame,
    df_stations: pd.DataFrame,
    mag_str: str = "3.5",
    file_name: Union[str, Path] = "tempfpfit.prt",
    latref_lonref: Optional[Union[List[float], tuple]] = None,
    remove_files: bool = True,
    max_threads: int = 1,
    use_tqdm: bool = True,
) -> Dict[str, Optional[List[Dict[str, Any]]]]:
    """
    Runs the FPfit inversion for a set of seismic events provided in a DataFrame.

    This function prepares the necessary FPfit input files, copies required executables to a temporary folder,
    and processes each event either sequentially or in parallel using a ThreadPoolExecutor. For each event,
    the `fpfit` function is called to generate inversion results (a list of dictionaries containing strike, dip,
    rake, and slip). After processing all events, the temporary folder is optionally removed.

    Parameters:
        df_real_test (pd.DataFrame): DataFrame containing event data. Each row should include fields such as
            'lat', 'lon', 'x', 'y', 'z', and polarity columns (e.g., 'pol_ST01', 'pol_ST02', ...).
        df_stations (pd.DataFrame): DataFrame containing station data. Must include a column 'staz' with station names,
            as well as station coordinates ('xstaz', 'ystaz', etc.).
        mag_str (str): Magnitude as a string (e.g., "3.5") to include in the FPfit input.
        file_name (Union[str, Path]): Base filename for the temporary FPfit input file.
        latref_lonref (Optional[Union[List[float], tuple]]): Optional reference latitude and longitude used for coordinate conversion.
        remove_files (bool): If True, the temporary folder and its files will be removed after processing.
        max_threads (int): Number of threads to use for parallel processing. If set to 1, processing is sequential.

    Returns:
        Dict[str, Optional[List[Dict[str, Any]]]]:
            A dictionary mapping event names (or generated event identifiers) to their FPfit inversion results.
            Each result is a list of dictionaries (with keys 'strike', 'dip', 'rake', and 'slip'),
            or None if the event was skipped or an error occurred.
    """
    # Save the current working directory to restore later.
    current_dir = Path().resolve()
    results: Dict[str, Optional[List[Dict[str, Any]]]] = {}

    # Build a list of polarity column names for each station in the stations DataFrame.
    pol_cols = [f"pol_{staz}" for staz in df_stations.staz]

    # Create a temporary folder for FPfit execution based on the provided file_name (without extension).
    if isinstance(file_name, str):
        folder_path = (library_directory / "".join(file_name.split(".")[:-1])).resolve()
    elif isinstance(file_name, Path):
        folder_path = file_name.parent / "fpfit" / "".join(file_name.name.split(".")[:-1])
        file_name = file_name.name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Copy necessary input files and executables to the temporary folder.
    shutil.copy(
        library_directory / "run_fpfit_input.inp", folder_path / "run_fpfit_input.inp"
    )
    shutil.copy(library_directory / "fpfit", folder_path / "fpfit")
    shutil.copy(library_directory / "fpplot", folder_path / "fpplot")

    # Process events in parallel if more than one thread is specified.
    if max_threads > 1:
        # Define the function to process a single event in a thread.
        def process_event_with_threads(
            i_event: int, event: pd.Series
        ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
            # Check if the event has sufficient polarity data; skip if not.
            if event[pol_cols].abs().sum() <= 5.0:
                print(f"Skipping event {i_event} with less or equal 5 polarities")
                result = None
            else:
                # Retrieve a thread-specific identifier from the current thread's name.
                thread_name = threading.current_thread().name.split("_")[-1]
                try:
                    # Call the fpfit function for the event, using a thread-specific file name.
                    
                    result = fpfit(
                        event,
                        df_stations,
                        mag_str,
                        pol_cols,
                        latref_lonref=latref_lonref,
                        file_name=folder_path / (thread_name + file_name),
                    )
                except Exception as e:
                    print(f"Error in event {i_event}: {e}")
                    result = None

            # Return the event's name if available; otherwise, generate an identifier.
            if "name" in event:
                return event["name"], result
            else:
                return f"event_{i_event}", result

        # Determine the desired ordering of events based on the "name" column if available.
        if "name" in df_real_test:
            ordered_events = df_real_test["name"].values
        else:
            ordered_events = [f"event_{i}" for i in df_real_test.index]

        # Use a ThreadPoolExecutor to process events concurrently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all events for processing.
            futures = {
                executor.submit(process_event_with_threads, i_event, event): i_event
                for i_event, event in df_real_test.iterrows()
            }

            if use_tqdm:
                iterations = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(df_real_test),
                    desc="FPYFIT (Parallel)",
                )
            else:
                iterations = concurrent.futures.as_completed(futures)
            # As each future completes, store its result in the 'results' dictionary.
            for future in iterations:
                key, result = future.result()  # Ottiene il risultato della funzione
                results[key] = result

        # Ensure that the results are ordered as in the original DataFrame.
        assert all([k in results.keys() for k in ordered_events]) and all(
            [f in ordered_events for f in results.keys()]
        ), "There are problems with the sort of the results"
        results = {k: results[k] for k in ordered_events if k in results}

    else:
        if use_tqdm:
            iterations = tqdm(
                df_real_test.iterrows(), total=len(df_real_test), desc="FPYFIT"
            )
        else:
            iterations = df_real_test.iterrows()
        # Sequential processing if max_threads == 1.
        for i_event, event in iterations:
            if event[pol_cols].abs().sum() <= 5.0:
                print(f"Skipping event {i_event} with less or equal 5 polarities")
                result = None
            else:
                try:
                    result = fpfit(
                        event,
                        df_stations,
                        mag_str,
                        pol_cols,
                        latref_lonref=latref_lonref,
                        file_name=folder_path / file_name,
                    )
                    # print("FPFIT Results:", result)
                except Exception as e:
                    print(f"Error in event {i_event}: {e}")
                    result = None

            if "name" in event:
                results[event["name"]] = result
            else:
                results[f"event_{i_event}"] = result

    # Clean up the temporary folder if requested.
    if remove_files:
        # print(f"Removing folder {folder_path}")
        shutil.rmtree(folder_path)

    # Restore the original working directory.
    os.chdir(current_dir)
    return results


# %%
