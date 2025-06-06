#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np

# %%

# Constants
R = 6378.388
E = 0.0033670033
PO180 = np.pi / 180  # Conversion from degrees to radians


def fr(t: np.floating):
    """
    Function to calculate the value of fr based on input t.
    """
    float_type = t.dtype.type
    return float_type(1.0) - float_type(E) * np.sin(t) ** float_type(2.0)


def gclc_check(
    zlatdeg: np.floating, zlondeg: np.floating, xlatdeg: np.ndarray, ylondeg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts geographical coordinates.
    """

    dtypes = [zlatdeg.dtype, zlondeg.dtype, xlatdeg.dtype, ylondeg.dtype]
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
    crlat = np.arctan(e * np.sin(float_type(2.0) * zlat) / fr(zlat))
    zgcl = zlat - crlat
    a = fr(zgcl)
    rho = r * a
    b = (e * np.sin(float_type(2.0) * zgcl) / a) ** float_type(2.0) + float_type(1.0)
    c = float_type(2.0) * e * np.cos(float_type(2.0) * zgcl) * a + (
        e * np.sin(float_type(2.0) * zgcl)
    ) ** float_type(2.0)
    cdist = c / (a ** float_type(2.0) * b) + float_type(1.0)

    # NOTE some values of cos(xlat - crlat) diverge from Fortran
    # output (in float32) by 1 in the integer representation.
    cos_factor = (
        np.cos((xlat - crlat).astype(np.float64)).astype(float_dtype)
        if float_dtype.itemsize == 4  # float32
        else np.cos(xlat - crlat)
    )

    yp = -rho * (zlon - ylon) * cos_factor
    xp = rho * (xlat - zlat) / cdist

    return xp, yp


# %%
