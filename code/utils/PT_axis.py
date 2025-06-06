#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%
import torch
import numpy as np
from typing import Union, Sequence, Tuple

# %%


class FocalMechanism:
    """
    Builds a series of focal mechanisms from an array of shape (n,3)
    containing [strike, dip, rake] in degrees for each entry.

    Calculates:
        - Fault normal:
            a = -sin(strike) * sin(dip)
            b =  cos(strike) * sin(dip)
            c = -cos(dip)
        - Slip vector:
            a = cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike)
            b = cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike)
            c = -sin(rake)*sin(dip)

    The P- and T-axes are then obtained as:
        - P-axis = normalize(slip_vector - fault_normal)
        - T-axis = normalize(slip_vector + fault_normal)

    Finally, the axes are converted to spherical coordinates (takeoff and azimuth)
    (the to_spherical_coords function returns (r, theta, phi), and theta is used as takeoff).
    """

    def __init__(self, sdr_array: Union[Sequence[float], np.ndarray]) -> None:
        """
        :param sdr_array: array of shape (n,3) or (3,) containing [strike, dip, rake] in degrees
        """
        # Store angles as a NumPy array of floats
        self.sdr = np.asarray(sdr_array, dtype=float)

        # If input is a single mechanism (1D), expand to 2D for consistency
        if sdr_array.ndim == 1:
            self.sdr = np.expand_dims(self.sdr, axis=0)
            self.initial_ndim = 1
        else:
            self.initial_ndim = 2

        # Validate input shape
        if self.sdr.shape[1] != 3:
            raise ValueError("L'array deve avere forma (n,3) o (3,)")

        # Convert angles from degrees to radians
        self.strike = np.deg2rad(self.sdr[:, 0])
        self.dip = np.deg2rad(self.sdr[:, 1])
        self.rake = np.deg2rad(self.sdr[:, 2])

    def get_fault_normal(self) -> np.ndarray:
        """
        Computes the fault normal vector for each mechanism.
        :return: array of shape (n,3)
        """
        phi = self.strike
        delta = self.dip
        # Components of the normal vector
        a = -np.sin(phi) * np.sin(delta)
        b = np.cos(phi) * np.sin(delta)
        c = -np.cos(delta)
        # Returns an array (n,3)
        return np.column_stack((a, b, c))

    def get_slip_vector(self) -> np.ndarray:
        """
        Computes the slip vector for each mechanism.
        :return: array of shape (n,3)
        """
        phi = self.strike
        delta = self.dip
        lam = self.rake
        # Components of the slip vector
        a = np.cos(lam) * np.cos(phi) + np.sin(lam) * np.cos(delta) * np.sin(phi)
        b = np.cos(lam) * np.sin(phi) - np.sin(lam) * np.cos(delta) * np.cos(phi)
        c = -np.sin(lam) * np.sin(delta)
        return np.column_stack((a, b, c))

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalizes a set of vectors row-wise to unit length.
        :param vectors: array of shape (n,3)
        :return: normalized vectors of shape (n,3)
        """
        norm: np.ndarray = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norm

    def get_PT(
        self,
    ) -> Tuple[
        Union[np.ndarray, Tuple[float, float]], Union[np.ndarray, Tuple[float, float]]
    ]:
        """
        Calculates the P- and T-axes for each focal mechanism and returns them in spherical coordinates.

        :return:
            P: array of shape (n,2) where each row is [takeoff, azimuth] in degrees for the P-axis,
                or a tuple (takeoff, azimuth) if n == 1.
            T: array of shape (n,2) where each row is [takeoff, azimuth] in degrees for the T-axis,
                or a tuple (takeoff, azimuth) if n == 1.
        """
        # Compute fault normal and slip vector
        fault_normal = self.get_fault_normal()
        slip_vector = self.get_slip_vector()

        # Compute P and T in Cartesian coordinates
        P_cart = self.normalize(slip_vector - fault_normal)
        T_cart = self.normalize(slip_vector + fault_normal)

        # Convert Cartesian vectors to spherical (r, theta, phi)
        _, theta_P, phi_P = self.to_spherical_coords(P_cart)
        _, theta_T, phi_T = self.to_spherical_coords(T_cart)

        # Convert angles to degrees: theta = takeoff, phi = azimuth
        P = np.column_stack((np.rad2deg(theta_P), np.rad2deg(phi_P)))
        T = np.column_stack((np.rad2deg(theta_T), np.rad2deg(phi_T)))

        # Squeeze if original input was 1D
        return P.squeeze(), T.squeeze()

    @staticmethod
    def to_spherical_coords(
        vectors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts an array of Cartesian vectors (n,3) to spherical coordinates:
            r     = vector magnitude,
            theta = inclination (angle from the Z-axis, in radians),
            phi   = azimuth (angle in the X-Y plane, in radians).

        :param vectors: array of shape (n,3)
        :return: tuple of arrays (r, theta, phi) each of shape (n,)
        """
        # Compute magnitude of each vector
        r = np.linalg.norm(vectors, axis=1)
        # Inclination angle: between vector and Z-axis
        theta = np.arccos(
            vectors[:, 2] / r
        )  # takeoff: angle between vector and horizontal plane
        # Azimuth angle in the X-Y plane
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        return r, theta, phi


# %%


class FocalMechanism_torch:
    """
    Builds a series of focal mechanisms from a tensor of shape (n,3)
    containing [strike, dip, rake] in degrees for each row.

    Calculates:
        - Fault normal:
            a = -sin(strike) * sin(dip)
            b =  cos(strike) * sin(dip)
            c = -cos(dip)
        - Slip vector:
            a = cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike)
            b = cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike)
            c = -sin(rake)*sin(dip)

    The P- and T-axes are then obtained by normalizing:
        - P-axis = slip_vector - fault_normal
        - T-axis = slip_vector + fault_normal

    Finally, the axes are converted to spherical coordinates via to_spherical_coords,
    extracting takeoff (theta) and azimuth (phi) in degrees.
    """

    def __init__(self, sdr_tensor: torch.Tensor) -> None:
        """
        :param sdr_tensor: tensor of shape (n,3) containing [strike, dip, rake] in degrees
        """
        # Ensure tensor is float32
        self.sdr = sdr_tensor.to(torch.float32)
        # Convert angles from degrees to radians
        self.strike = torch.deg2rad(self.sdr[:, 0])
        self.dip = torch.deg2rad(self.sdr[:, 1])
        self.rake = torch.deg2rad(self.sdr[:, 2])

    def get_fault_normal(self) -> torch.Tensor:
        """
        Computes the fault normal vector for each mechanism.
        :return: tensor of shape (n,3)
        """
        phi = self.strike
        delta = self.dip
        # Components of the normal vector
        a = -torch.sin(phi) * torch.sin(delta)
        b = torch.cos(phi) * torch.sin(delta)
        c = -torch.cos(delta)
        # Returns a tensor (n,3)
        return torch.stack((a, b, c), dim=1)

    def get_slip_vector(self) -> torch.Tensor:
        """
        Computes the slip vector for each mechanism.
        :return: tensor of shape (n,3)
        """
        phi = self.strike
        delta = self.dip
        lam = self.rake
        # Components of the slip vector
        a = torch.cos(lam) * torch.cos(phi) + torch.sin(lam) * torch.cos(
            delta
        ) * torch.sin(phi)
        b = torch.cos(lam) * torch.sin(phi) - torch.sin(lam) * torch.cos(
            delta
        ) * torch.cos(phi)
        c = -torch.sin(lam) * torch.sin(delta)
        return torch.stack((a, b, c), dim=1)

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a set of vectors row-wise to unit length.
        :param vectors: tensor of shape (n,3)
        :return: normalized tensor of shape (n,3)
        """
        norm = torch.norm(vectors, dim=1, keepdim=True)
        return vectors / norm

    def get_PT(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the P- and T-axes for each focal mechanism and returns them in spherical coordinates.

        :return:
            P: tensor of shape (n,2) where each row is [takeoff, azimuth] in degrees for the P-axis.
            T: tensor of shape (n,2) where each row is [takeoff, azimuth] in degrees for the T-axis.
        """
        # Compute fault normal and slip vector
        fault_normal = self.get_fault_normal()
        slip_vector = self.get_slip_vector()

        # Compute P and T in Cartesian coordinates
        P_cart = self.normalize(slip_vector - fault_normal)
        T_cart = self.normalize(slip_vector + fault_normal)

        # Convert Cartesian vectors to spherical (r, theta, phi)
        _, theta_P, phi_P = self.to_spherical_coords(P_cart)
        _, theta_T, phi_T = self.to_spherical_coords(T_cart)

        # Convert angles to degrees: theta = takeoff, phi = azimuth
        P = torch.stack((torch.rad2deg(theta_P), torch.rad2deg(phi_P)), dim=1)
        T = torch.stack((torch.rad2deg(theta_T), torch.rad2deg(phi_T)), dim=1)
        return P, T

    @staticmethod
    def to_spherical_coords(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a tensor of vectors of shape (n,3) to spherical coordinates:
            r     = magnitude of the vector,
            theta = inclination (angle from the Z-axis, in radians),
            phi   = azimuth (angle in the X-Y plane, in radians).

        :param vectors: tensor of shape (n,3)
        :return: tuple of tensors (r, theta, phi) each of shape (n,)
        """
        # Compute magnitude of each vector
        r = torch.norm(vectors, dim=1)
        # Inclination angle: between vector and Z-axis
        theta = torch.acos(
            torch.clamp(
                vectors[:, 2] / r, min=-1.0, max=1.0
            )  # theta: angle between vector and Z-axis
        )
        # Azimuth angle in the X-Y plane
        phi = torch.atan2(vectors[:, 1], vectors[:, 0])
        return r, theta, phi


# %%
