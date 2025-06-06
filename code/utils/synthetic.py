#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import numpy as np
import torch
from pathlib import Path

# Import the takeoff function from a local module
from .fsubroutines.azih import takeoff as takeoff_f

# %%


def ensure_float_dtype(dtype: str | torch.dtype | None = None) -> torch.dtype:
    """
    Ensure the given data type is a valid floating-point PyTorch dtype.

    Args:
        dtype (str | torch.dtype | None, optional): The dtype as a string, PyTorch dtype, or None.
            - If None, defaults to "float32".
            - If a string, it is converted to a PyTorch dtype.
            - If already a PyTorch dtype, it is used directly.

    Returns:
        torch.dtype: The validated floating-point PyTorch dtype.

    Raises:
        ValueError: If the dtype is not a floating-point type.
    """
    if dtype is None:
        dtype = "float32"  # Default to float32 if no dtype is provided
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)  # Convert string to PyTorch dtype
    if not dtype.is_floating_point:
        raise ValueError(
            "dtype must be a floating point type"
        )  # Ensure dtype is floating point
    return dtype


def ensure_tensor_dtype(
    tensors: list[torch.Tensor], dtype: torch.dtype
) -> list[torch.Tensor]:
    """
    Ensure all tensors in the input list have the specified dtype.

    Args:
        tensors (list[torch.Tensor]): A list of input tensors.
        dtype (torch.dtype): The target floating-point dtype.

    Returns:
        list[torch.Tensor]: A list of tensors converted to the specified dtype.

    Raises:
        ValueError: If any tensor is not a floating-point type.
    """
    out_tensors = []
    for t in tensors:
        dt = t.dtype
        if not dt.is_floating_point:
            raise ValueError(
                "Input types must be floating point"
            )  # Ensure input tensors are floating point

        # Convert tensor dtype if needed, otherwise keep it unchanged
        out_tensors.append(t if dt == dtype else t.to(dtype))

    return out_tensors


# %%


def read_velmod_data(
    filepath: Path, dtype: str | torch.dtype | None = None
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Reads velocity model data from a file and returns the reference latitude,
    reference longitude, and depth-velocity pairs.

    Args:
        filepath (Path): Path to the velocity model data file.
        dtype (str | torch.dtype | None, optional): Desired floating-point dtype for tensors.
            Defaults to None, which is converted to "float32".

    Returns:
        tuple:
            - latref (float): Reference latitude.
            - lonref (float): Reference longitude.
            - v (torch.Tensor): 1D tensor of velocity values.
            - d (torch.Tensor): 1D tensor of depth values.
    """

    # Ensure the dtype is a valid floating point type
    dtype = ensure_float_dtype(dtype)

    # Initialize lists to store velocity and depth values
    v_l = []
    d_l = []

    # Open and read the velocity model file
    with open(filepath, "r") as file:
        # Read the first line containing reference latitude, longitude, and number of layers
        latref_s, lonref_s, nl_s = next(file).split()
        latref, lonref = float(latref_s), float(lonref_s)
        nl = int(nl_s)  # Number of depth-velocity layers

        # Read the depth-velocity data for each layer
        for _ in range(nl):
            vl, dl = [float(x) for x in next(file).split()]
            v_l.append(vl)
            d_l.append(dl)

    # Convert lists to PyTorch tensors with the specified dtype
    v = torch.tensor(v_l, dtype=dtype)
    d = torch.tensor(d_l, dtype=dtype)
    return latref, lonref, v, d


def read_stations_data(
    filepath: Path, dtype: str | torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads seismic station data from a file and returns station names, latitudes, longitudes, and elevations.

    Args:
        filepath (Path): Path to the stations data file.
        dtype (str | torch.dtype | None, optional): Desired floating-point dtype for tensors.
            Defaults to None, which is converted to "float32".

    Returns:
        tuple:
            - staz (list[str]): List of station names.
            - lats (torch.Tensor): 1D tensor of latitude values.
            - lons (torch.Tensor): 1D tensor of longitude values.
            - elev (torch.Tensor): 1D tensor of elevation values.
    """

    # Ensure the dtype is a valid floating point type
    dtype = ensure_float_dtype(dtype)

    # Initialize lists to store station data
    staz_l = []  # List for station names
    lats_l = []  # List for latitude values
    lons_l = []  # List for longitude values
    elev_l = []  # List for elevation values

    # Open and read the stations data file
    with open(filepath, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 4:  # Ensure the line contains four values
                break
            staz_l.append(parts[0])  # Station name
            lons_l.append(float(parts[1]))  # Longitude
            lats_l.append(float(parts[2]))  # Latitude
            elev_l.append(float(parts[3]))  # Elevation

    # Convert numerical lists to PyTorch tensors
    lats = torch.tensor(lats_l, dtype=dtype)
    lons = torch.tensor(lons_l, dtype=dtype)
    elev = torch.tensor(elev_l, dtype=dtype)

    return staz_l, lats, lons, elev


# %%


def fr(t: torch.Tensor, dtype: str | torch.dtype | None = None) -> torch.Tensor:
    """
    Computes a correction factor for geodetic calculations.

    Args:
        t (torch.Tensor): Input angle in radians.
        dtype (str | torch.dtype | None, optional): Target floating-point dtype.

    Returns:
        torch.Tensor: The computed correction factor.
    """
    t = ensure_tensor_dtype([t], dtype)[0]  # Ensure tensor has correct dtype

    E = 0.0033670033  # Earth's ellipticity factor
    return (1.0 - E * (torch.sin(t) ** 2.0)).to(dtype=dtype)


def gclc(
    zlatdeg: float,
    zlondeg: float,
    xlatdeg: torch.Tensor,
    ylondeg: torch.Tensor,
    dtype: str | torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts geographical coordinates (latitude, longitude) into Cartesian x-y coordinates.

    The coordinate system is centered at a reference latitude and longitude.

    Args:
        zlatdeg (float): Reference latitude in degrees.
        zlondeg (float): Reference longitude in degrees.
        xlatdeg (torch.Tensor): Input latitudes in degrees.
        ylondeg (torch.Tensor): Input longitudes in degrees.
        dtype (str | torch.dtype | None, optional): Target floating-point dtype.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x and y coordinates in the Cartesian system.
    """

    # Constants
    R = 6378.388  # Earth's radius in km
    E = 0.0033670033  # Earth's ellipticity factor

    # Ensure dtype is a valid floating point type
    dtype = ensure_float_dtype(dtype)

    # Convert input latitude and longitude tensors to the correct dtype
    xlatdeg, ylondeg = ensure_tensor_dtype([xlatdeg, ylondeg], dtype)

    # Convert input degrees to radians
    zlat = torch.deg2rad(torch.tensor(zlatdeg))  # Reference latitude in radians
    zlon = torch.deg2rad(torch.tensor(zlondeg))  # Reference longitude in radians
    xlat = torch.deg2rad(xlatdeg)  # Convert input latitudes
    ylon = torch.deg2rad(ylondeg)  # Convert input longitudes

    # Compute geodetic correction factor for reference latitude
    crlat = torch.arctan(E * torch.sin(2.0 * zlat) / fr(zlat))
    zgcl = zlat - crlat  # Corrected latitude

    # Compute local radius of curvature
    a = fr(zgcl)
    rho = R * a  # Scaled Earth's radius for current latitude
    b = (E * torch.sin(2.0 * zgcl) / a) ** 2.0 + 1.0
    c = 2.0 * E * torch.cos(2.0 * zgcl) * a + (E * torch.sin(2.0 * zgcl)) ** 2.0
    cdist = c / (a**2.0 * b) + 1.0  # Distance scaling factor

    # Compute correction factor for float precision compatibility
    # Ensures consistency with original Fortran calculations
    # NOTE some values of cos(xlat - crlat) diverge from Fortran
    # output (in float32) by 1 in binary representation.
    cos_factor = (
        torch.cos((xlat - crlat).to(torch.float64))
        if dtype.itemsize < 8  # Use double precision for float32
        else torch.cos(xlat - crlat)
    )

    # Compute Cartesian coordinates
    yp = -rho * (zlon - ylon) * cos_factor  # y coordinate (longitude difference scaled)
    xp = rho * (xlat - zlat) / cdist  # x coordinate (latitude difference scaled)

    return xp.to(dtype), yp.to(dtype)


# %%


def azimuth(
    xstaz: torch.Tensor,
    ystaz: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the azimuthal angle between seismic stations and a source location.

    Args:
        xstaz (torch.Tensor): X-coordinates of the stations.
        ystaz (torch.Tensor): Y-coordinates of the stations.
        xs (torch.Tensor): X-coordinates of the source(s).
        ys (torch.Tensor): Y-coordinates of the source(s).

    Returns:
        torch.Tensor: Azimuth angles in degrees, adjusted to the [0, 360] range.

    Raises:
        ValueError: If any input tensor is not of a floating-point type.
    """

    # Validate that all input tensors have floating-point data types
    dtypes = [xstaz.dtype, ystaz.dtype, xs.dtype, ys.dtype]

    if any(not dt.is_floating_point for dt in dtypes):
        raise ValueError("strikes: Input types must be floating point")

    # Determine the highest precision floating-point type
    float_type: type = xstaz.dtype
    if len(set(dtypes)) != 1:
        print("Array types not matching. Promoting to the highest precision")
        float_type = max(dtypes, key=lambda x: x.itemsize)

    # Compute azimuth angle in degrees
    phirdeg = torch.rad2deg(
        torch.arctan2(xstaz[None, :] - xs[:, None], ystaz[None, :] - ys[:, None])
    )

    # Adjust azimuth angle to be in the range [0, 360] degrees
    phirdeg = phirdeg - 180.0 * (
        -1 + torch.copysign(torch.tensor(1.0).to(dtype=float_type), phirdeg)
    )
    phirdeg %= 360.0  # Ensures azimuth remains within the [0, 360] range

    return phirdeg.to(dtype=float_type)


def takeoff(
    vv: torch.Tensor,
    dd: torch.Tensor,
    depi: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the takeoff angle for seismic waves using a velocity-depth model.

    Args:
        vv (torch.Tensor): Velocity model as a 1D tensor.
        dd (torch.Tensor): Depths corresponding to the velocity model.
        depi (torch.Tensor): Epicentral distances as a 2D tensor (batch_size, num_stations).
        z (torch.Tensor): Source depths.

    Returns:
        torch.Tensor: Computed takeoff angles.

    Raises:
        AssertionError: If the number of source depths does not match the number of distances.
    """

    arr_type = vv.dtype  # Store dtype of velocity tensor
    device = vv.device  # Store device (cpu/cuda) of velocity tensor
    nl = len(vv)  # Number of layers in velocity model

    # Ensure that the number of depths matches the number of distances
    assert len(depi) == len(z), "Mismatch: Number of source depths must match number of distances"

    # Initialize empty tensors for velocity (V) and depth (D) with default size 119
    V = torch.empty(119, dtype=torch.float32)
    D = torch.empty(119, dtype=torch.float32)

    # Fill the first `nl` elements with actual velocity and depth values
    V[:nl] = vv
    D[:nl] = dd

    # Convert tensors to NumPy for compatibility with the Fortran routine
    V = V.numpy()
    D = D.numpy()

    # Convert distances and depths to Fortran-contiguous NumPy arrays
    distm = np.asfortranarray(depi.cpu().numpy().astype("float32"))
    zak = z.cpu().numpy().astype("float32")

    # Call the Fortran-compiled takeoff function
    out_np, _ = takeoff_f(V, D, nl, depi.shape[1], len(z), zak, distm)

    # Convert the output back to a PyTorch tensor with the original dtype and device
    out = torch.tensor(out_np).to(arr_type).to(device)
    return out


# %%


def compute_probs(vett: torch.Tensor) -> torch.Tensor:
    """
    Compute probability distribution by normalizing each row in a tensor.

    Args:
        vett (torch.Tensor): Input tensor of shape (N, M), where N is the batch size
                            and M is the number of probabilities per batch.

    Returns:
        torch.Tensor: A tensor of the same shape with normalized probabilities per row.
    """
    return vett / vett.sum(axis=1, keepdim=True)


# %%


def trigo(dip: torch.Tensor, slp: torch.Tensor) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute trigonometric functions used in moment tensor decomposition.

    Args:
        dip (torch.Tensor): Dip angles in radians.
        slp (torch.Tensor): Slip angles in radians.

    Returns:
        tuple[torch.Tensor]: Computed trigonometric coefficients.
    """

    r1 = 1.0
    r2 = 2.0
    
    # Compute basic trigonometric values
    sdi = torch.sin(dip)  # sin(dip)
    cdi = torch.cos(dip)  # cos(dip)
    ssl = torch.sin(slp)  # sin(slip)
    csl = torch.cos(slp)  # cos(slip)

    # Compute required values for tensor calculations
    sdi2 = r2 * sdi * cdi
    cdi2 = r1 - r2 * sdi * sdi
    f1 = csl * sdi
    f2 = -csl * cdi
    f3 = ssl * sdi2
    f4 = -f3
    f5 = ssl * cdi2
    f6 = -f3 / r2
    return f1, f2, f3, f4, f5, f6


def radgam(
    azi: torch.Tensor,
    aih: torch.Tensor,
    strk: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    f4: torch.Tensor,
    f5: torch.Tensor,
    f6: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute amplitude components.

    Args:
        azi (torch.Tensor): Azimuth angles in radians.
        aih (torch.Tensor): Incidence angles in radians.
        strk (torch.Tensor): Strike angles in radians.
        f1, f2, f3, f4, f5, f6 (torch.Tensor): Trigonometric coefficients computed from `trigo()`.

    Returns:
        tuple[torch.Tensor]: (cp, csv, csh) amplitude components.
    """

    r1 = 1.0
    r2 = 2.0

    # Compute trigonometric values
    sai = torch.sin(aih)  # sin(incidence angle)
    cai = torch.cos(aih)  # cos(incidence angle)
    sacai = sai * cai  # sin(incidence) * cos(incidence)
    sai2 = r2 * sacai  # 2 * sin(incidence) * cos(incidence)
    sai22 = sai * sai  # sin²(incidence)
    cai2 = r1 - r2 * sai22  # cos²(incidence) - sin²(incidence)
    
    # Compute azimuthal differences
    dfi = azi - strk
    sdf = torch.sin(dfi)  # sin(azimuth - strike)
    cdf = torch.cos(dfi)  # cos(azimuth - strike)
    sdf2 = r2 * sdf * cdf  # 2 * sin(azimuth - strike) * cos(azimuth - strike)
    sdf22 = sdf * sdf  # sin²(azimuth - strike)
    cdf2 = r1 - r2 * sdf22  # cos²(azimuth - strike) - sin²(azimuth - strike)

    # Compute amplitude components
    g1 = f1 * sdf2 + f4 * sdf22
    g2 = f2 * cdf + f5 * sdf
    cp = sai22 * g1 + sai2 * g2 + f3 * cai * cai  # P-wave amplitude
    csv = cai2 * g2 + sacai * (g1 - f3) # SV-wave amplitude
    csh = cai * (f5 * cdf - f2 * sdf) + sai * (f1 * cdf2 + f6 * sdf2) # SH-wave amplitude

    return cp, csv, csh


def radpar(
    az: torch.Tensor,
    strikediprake: torch.Tensor,
    ih: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute seismic amplitude parameters.

    Args:
        az (torch.Tensor): Azimuth angles in degrees.
        strikediprake (torch.Tensor): Tensor containing strike, dip, and rake angles in degrees.
        ih (torch.Tensor): Incidence angles in degrees.

    Returns:
        tuple[torch.Tensor]: (cp, csv, csh) amplitude components.
    """

    # Convert input angles from degrees to radians
    strkdiprslp = torch.deg2rad(strikediprake)
    strk = strkdiprslp[..., 0:1]  # Extract strike angle
    dipr = strkdiprslp[..., 1:2]  # Extract dip angle
    slp = strkdiprslp[..., 2:3]  # Extract rake angle

    azi = torch.deg2rad(az)  # Convert azimuth to radians
    aih = torch.deg2rad(ih)  # Convert incidence angle to radians

    # Compute trigonometric coefficients
    f1, f2, f3, f4, f5, f6 = trigo(dipr, slp)
    
    # Compute amplitude components
    cp, csv, csh = radgam(azi, aih, strk, f1, f2, f3, f4, f5, f6)

    return cp, csv, csh


# %%
