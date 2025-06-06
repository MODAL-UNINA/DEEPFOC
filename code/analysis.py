#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %% [markdown]
# IMPORTS
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from strec.kagan import get_kagan_angle
import seaborn as sns
from tqdm.auto import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio

from utils.gen_utils import read_velmod_data
from utils.online_dataset import FocalOnlineDataset
from utils.loss import KaganAngle
from utils.model import AmplitudePolaritiesModel
from utils.utils import myshow

kagan_angle = KaganAngle()

scripts_dir = Path(os.getcwd()).resolve()
root_dir = scripts_dir.parent
save_folder = "Plots"
save_dir = root_dir / "results" / save_folder
save_dir.mkdir(parents=True, exist_ok=True)
data_dir = root_dir / "data" 
input_dir = data_dir / "INPUT"

device = "cpu"

# %%

# Plot: Beta distribution
# This cell generates a grid of Beta distributions with varying modes and
# total variation parameters (k).

modes = [0.0, 0.3, 0.7, 1.0]
ks = [3, 5, 10]
# Plot: Beta distribution
fig, axes = plt.subplots(nrows=len(ks), ncols=len(modes), figsize=(20, 12), sharex=True)
for i, k in enumerate(ks):
    for j, mode in enumerate(modes):
        # Compute alpha and beta
        alpha = mode * (k - 2) + 1
        beta = (1 - mode) * (k - 2) + 1
        dist = torch.distributions.Beta(alpha, beta)
        points = dist.sample((1000,))

        ax = axes[i, j]
        # Histplot with Matplotlib
        sns.histplot(
            points.numpy(),
            kde=True,
            stat="density",
            ax=ax,
            alpha=0.6,
        )
        ax.set_title(f"k={k}, m={mode}", fontsize=14)

        if i == len(modes) - 1:
            ax.set_xlabel("Value", fontsize=12)
        if j == 0:
            ax.set_ylabel("Density", fontsize=12)

plt.tight_layout()
plt.savefig(save_dir / "beta_distribution.png", dpi=300, bbox_inches="tight")
myshow()


# %%

# This cell generates synthetic seismic event data, focusing on earthquake hypocenter
# locations (X, Y, Z coordinates) and focal mechanisms (Strike, Dip, Rake angles).
# It then visualizes the distributions of these parameters using 3D scatter plots
# for hypocenters and histograms for focal mechanism angles, saving the plots
# as PNG files.

gen = FocalOnlineDataset(
    xyz_boundary=[-4.0, 4.0, -4.0, 3.1, 2.0, 3.0],
    sdr_boundary=[0, 360, 0, 90, -180, 180],
    xyz_steps=[0.1, 0.1, 0.05],
    sdr_steps=[1, 1, 1],
    exclude_strike=[360],
    exclude_dip=[0, 90],
    exclude_rake=[-180, 180],
    velmod_file="velmod_campi_flegrei.dat",
    stations_file="stations_campi_flegrei.dat",
    input_dir=input_dir,
    batch_size=int(2**11),
    generator=torch.Generator(device=device),
    device=device,
    xyz_translation_step_percs=None,
)

data = gen.gen_batch()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    data["XYZ"][:, 0],
    data["XYZ"][:, 1],
    data["XYZ"][:, 2],
    c=data["XYZ"][:, 2],
    cmap="viridis",
    s=20,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.5,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])
# Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("Z Value")
ax.set_title("XYZ Distribution")
plt.savefig(save_dir / "xyz_distribution.png", dpi=300, bbox_inches="tight")
myshow()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    data["XYZ"][:, 0],  # X
    data["XYZ"][:, 1],  # Y
    data["XYZ"][:, 2],  # Z
    c=data["XYZ"][:, 2],
    cmap="plasma",
    s=20,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.5,
)

ax.view_init(elev=10, azim=80)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])

cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("Z Value")
ax.set_title("XYZ Distribution")

plt.savefig(save_dir / "xyz_distribution.png", dpi=300, bbox_inches="tight")
myshow()

plt.style.use("seaborn-v0_8-white")


def plot_hypocenter_3d_fette_piani(x, y, z, num_fette=8):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    z_min, z_max = np.min(z), np.max(z)
    fette_edges = np.linspace(z_min, z_max, num_fette + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, num_fette)[::-1])

    for i in range(num_fette):
        mask = (z >= fette_edges[i]) & (z < fette_edges[i + 1])
        ax.scatter(
            x[mask],
            y[mask],
            -z[mask],
            c=colors[i].reshape(1, -1),
            alpha=0.8,
            s=10,
            edgecolors="w",
            linewidth=0.2,
            label=f"{fette_edges[i]:.2f}-{fette_edges[i+1]:.2f} km",
        )

        xx, yy = np.meshgrid([np.min(x), np.max(x)], [np.min(y), np.max(y)])
        zz = -np.ones_like(xx) * fette_edges[i + 1]
        ax.plot_surface(xx, yy, zz, color="grey", alpha=0.08, edgecolor="none")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Depth (km)")
    ax.set_title("3D Hypocenters with Depth Slices", fontsize=14, pad=15)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_zlim(-z_max, -z_min)
    ax.legend(title="Depth slices", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_dir / "3d_hypocenters_fette_piani.png", dpi=300)
    myshow()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Strike angle
axes[0].hist(data["SDR"][:, 0], bins=36, color="#6495ED", alpha=0.9)
axes[0].set_xlabel("Strike Angle (°)", fontsize=18)

# Dip angle
axes[1].hist(data["SDR"][:, 1], bins=18, color="#DB7093", alpha=0.9)
axes[1].set_xlabel("Dip Angle (°)", fontsize=18)

# Rake angle
axes[2].hist(data["SDR"][:, 2], bins=18, color="#8FBC8F", alpha=0.9)
axes[2].set_xlabel("Rake Angle (°)", fontsize=18)

for ax in axes:
    ax.set_ylabel("Frequency", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


plt.tight_layout()
plt.savefig(
    save_dir / "focal_mechanism_distributions.png", dpi=300, bbox_inches="tight"
)
myshow()

# strike Distribution
plt.figure(figsize=(8, 6))
plt.hist(data["SDR"][:, 0], bins=18, color="#6495ED", alpha=0.9)
plt.xlabel("Strike Angle (°)", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "strike_distribution.png", dpi=300)
myshow()

# dip Distribution
plt.figure(figsize=(8, 6))
plt.hist(data["SDR"][:, 1], bins=18, color="#DB7093", alpha=0.9)
plt.xlabel("Dip Angle (°)", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "dip_distribution.png", dpi=300)
myshow()

# rake Distribution
plt.figure(figsize=(8, 6))
plt.hist(data["SDR"][:, 2], bins=18, color="#8FBC8F", alpha=0.9)
plt.xlabel("Rake Angle (°)", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "rake_distribution.png", dpi=300)
myshow()


# %%

# This cell performs a detailed visualization of the generated earthquake
# hypocenter locations (X, Y, Z coordinates). It generates histograms to show
# the marginal distribution of each coordinate and creates 2D scatter plots
# (XY, XZ, YZ) using both Matplotlib and Plotly (go.Scatter) to show their
# pairwise distributions. Finally, it re-plots the 3D hypocenters with depth
# slices for a comprehensive spatial overview, saving all plots as PNG files.

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Strike angle
axes[0].hist(data["XYZ"][:, 0], bins=27, color="#4C72B0", alpha=0.9)
axes[0].set_title("X Coordinate")

# Dip angle
axes[1].hist(data["XYZ"][:, 1], bins=24, color="#DD8452", alpha=0.9)
axes[1].set_title("Y Coordinate")

# Rake angle
axes[2].hist(data["XYZ"][:, 2], bins=21, color="#55A868", alpha=0.9)
axes[2].set_title("Z Coordinate")

for ax in axes:
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Coordinate Value")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.suptitle("Distribution of Hypocenters Coordinates", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig(save_dir / "xyz_distributions.png", dpi=300, bbox_inches="tight")
myshow()


trace = go.Scatter(
    x=data["XYZ"][:, 0],
    y=data["XYZ"][:, 1],
    mode="markers",
)

fig = go.Figure(trace)
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    margin=dict(l=5, r=30, t=30, b=5),
    paper_bgcolor="white",
    xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
    yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
)

fig.write_image(
    save_dir / "xy_distribution.png", engine="kaleido", width=800, height=600, scale=2.0
)
myshow(fig=fig)

trace = go.Scatter(
    x=data["XYZ"][:, 0],
    y=data["XYZ"][:, 2],
    mode="markers",
)

fig = go.Figure(trace)
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Z",
    margin=dict(l=5, r=30, t=30, b=5),
    paper_bgcolor="white",
    xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
    yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
)
fig.write_image(
    save_dir / "xz_distribution.png", engine="kaleido", width=800, height=600, scale=2.0
)
myshow(fig=fig)

trace = go.Scatter(
    x=data["XYZ"][:, 1],
    y=data["XYZ"][:, 2],
    mode="markers",
)

fig = go.Figure(trace)
fig.update_layout(
    xaxis_title="Y",
    yaxis_title="Z",
    margin=dict(l=5, r=30, t=30, b=5),
    paper_bgcolor="white",
    xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
    yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
)
fig.write_image(
    save_dir / "yz_distribution.png", engine="kaleido", width=800, height=600, scale=2.0
)
myshow(fig=fig)


# XY Distribution
plt.figure(figsize=(8, 6))
plt.scatter(data["XYZ"][:, 0], data["XYZ"][:, 1], s=15)
plt.xlabel("X", fontsize=18)
plt.ylabel("Y", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "xy_distribution.png", dpi=300)
myshow()

# XZ Distribution
plt.figure(figsize=(8, 6))
plt.scatter(data["XYZ"][:, 0], data["XYZ"][:, 2], s=15)
plt.xlabel("X", fontsize=18)
plt.ylabel("Z", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "xz_distribution.png", dpi=300)
myshow()

# YZ Distribution
plt.figure(figsize=(8, 6))
plt.scatter(data["XYZ"][:, 1], data["XYZ"][:, 2], s=15)
plt.xlabel("Y", fontsize=18)
plt.ylabel("Z", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "yz_distribution.png", dpi=300)
myshow()


# Plot: Hypocenter distributions
plot_hypocenter_3d_fette_piani(
    x=to_numpy(data["XYZ"][:, 0]),
    y=to_numpy(data["XYZ"][:, 1]),
    z=to_numpy(data["XYZ"][:, 2]),
)

# %%

# This cell reads the 1D velocity model for the Campi Flegrei area from a
# specified file and then visualizes it. It generates a step plot showing
# how seismic wave velocity changes with depth, saving the plot as a PNG file.

latref, lonref, v_cf, d_cf = read_velmod_data(
    input_dir / "velmod_campi_flegrei.dat", np.dtype("float64")
)

z_cf = np.array(d_cf)
fig, ax = plt.subplots()
ax.step(v_cf, z_cf, where="post", label="V Campi Flegrei", linewidth=2)
ax.set_ylim(z_cf.max(), z_cf.min())  # profondità verso il basso
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Deep (km)")
ax.grid(True)
ax.legend()
plt.savefig(save_dir / "1d_velmodel_cf.png", dpi=300, bbox_inches="tight")
myshow()


# %%
# This cell generates synthetic seismic data with a fixed hypocenter location
# but varying focal mechanisms (strike, dip, rake). It then calculates the Kagan
# angle for each generated focal mechanism relative to a reference angle.
# The primary goal is to visualize how the P-wave radiation pattern (Cp values)
# varies for events with small Kagan angles (less than 10 or 20 degrees)
# compared to an original/reference event, saving these comparative plots.

# Fixed Hypocenter
fixed_hypo = FocalOnlineDataset(
    xyz_boundary=[0.0, 0.0, 0.0, 0.0, 2.5, 2.5],
    sdr_boundary=[0, 360, 0, 90, -180, 180],
    xyz_steps=[0.1, 0.1, 0.05],
    sdr_steps=[2.5, 2.5, 2.5],
    exclude_strike=[360],
    exclude_dip=[0, 90],
    exclude_rake=[-180, 180],
    velmod_file="velmod_campi_flegrei.dat",
    stations_file="stations_campi_flegrei.dat",
    input_dir=input_dir,
    batch_size=int(2**11),
    generator=torch.Generator(device=device),
    device=device,
    xyz_translation_step_percs=None,
)

data = fixed_hypo.gen_data()

angle_ref = [0.0, 45.0, 0.0]

kagan_values = [get_kagan_angle(*f, *angle_ref) for f in tqdm(data["SDR"].numpy())]
df = pd.DataFrame(
    {
        "Strike": data["SDR"].numpy()[:, 0],
        "Dip": data["SDR"].numpy()[:, 1],
        "Rake": data["SDR"].numpy()[:, 2],
        "Kagan": kagan_values,
        "Cp": data["Cp"].tolist(),
    }
)

# Cp Kagan less than 10
original = df[df["Kagan"] == df["Kagan"].min()]
df_less_10 = df[df["Kagan"] < 10]
df_less_20 = df[df["Kagan"] < 20]

pio.renderers.default = "vscode"

df_less_10.drop(index=original.index, inplace=True)
df_less_20.drop(index=original.index, inplace=True)

# Plot: Cp for Kagan < 10
fig = plt.figure(figsize=(8, 6))
x = np.arange(0, len(df_less_10["Cp"].values[0]))
for i in range(df_less_10.shape[0]):
    y = df_less_10["Cp"].values[i]
    plt.plot(
        x,
        y,
        color="royalblue",
        alpha=0.4,
        label="Samples (Kagan < 10°)" if i == 0 else "",
    )
plt.plot(
    np.array(original["Cp"].values[0]),
    color="red",
    alpha=1.0,
    linestyle="--",
    label="Original",
)
plt.xlabel("Station Index", fontsize=18)
plt.ylabel("Cp", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Station Index")
plt.ylabel("Cp")
plt.grid()
plt.legend(
    fontsize=14,
    loc="upper right",
    bbox_to_anchor=(1.01, 1.01),
    borderaxespad=0.0,
    frameon=False,
)
plt.savefig(save_dir / "cp_kagan_less_10.png", dpi=300, bbox_inches="tight")
myshow()

# Plot: Cp for Kagan < 20
fig = plt.figure(figsize=(8, 6))
x = np.arange(0, len(df_less_20["Cp"].values[0]))
for i in range(df_less_20.shape[0]):
    y = df_less_20["Cp"].values[i]
    plt.plot(
        x,
        y,
        color="cadetblue",
        alpha=0.4,
        label="Samples (Kagan < 20°)" if i == 0 else "",
    )
plt.plot(
    np.array(original["Cp"].values[0]),
    color="red",
    alpha=1.0,
    linestyle="--",
    label="Original",
)
plt.xlabel("Station Index", fontsize=18)
plt.ylabel("Cp", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(
    fontsize=14,
    loc="upper right",
    bbox_to_anchor=(1.01, 1.01),
    borderaxespad=0.0,
    frameon=False,
)
plt.savefig(save_dir / "cp_kagan_less_20.png", dpi=300, bbox_inches="tight")
myshow()

# %%

# This cell focuses on simulating and visualizing the preprocessing effects
# on synthetic seismic data, specifically on P-wave radiation patterns (Cp values)
# and amplitudes. It generates a single seismic event (repeated multiple times),
# applies preprocessing steps (zeroing and adding noise), and then plots
# the preprocessed data against the original data using both line and scatter
# plots to highlight the changes induced by preprocessing. Plots are saved as PNG files.

fixed_hypo = FocalOnlineDataset(
    xyz_boundary=[0.0, 0.0, 0.0, 0.0, 2.5, 2.5],
    sdr_boundary=[0, 360, 0, 90, -180, 180],
    xyz_steps=[0.1, 0.1, 0.05],
    sdr_steps=[2.5, 2.5, 2.5],
    exclude_strike=[360],
    exclude_dip=[0, 90],
    exclude_rake=[-180, 180],
    velmod_file="velmod_campi_flegrei.dat",
    stations_file="stations_campi_flegrei.dat",
    input_dir=input_dir,
    batch_size=1,
    generator=torch.Generator(device=device),
    device=device,
    xyz_translation_step_percs=None,
)

single_data = fixed_hypo.gen_batch()
for key, value in single_data.items():
    value = value.repeat(100, 1)
    single_data[key] = value
preprocessing = AmplitudePolaritiesModel(
    n_stations=single_data["az"].shape[1],
    xyz_boundary=fixed_hypo.xyz_boundary,
    scaling_range=[0, 1, 0, 1, 0, 1],
).preprocessing

# Preprocessing statistics
preprocessed_data = preprocessing(single_data, [0.7, 0.9], normal_noise=0.25)
preprocessed_data["Cp_prepro"] = preprocessed_data["Polarities"] * torch.where(
    preprocessed_data["Amplitudes"] == -1, 0, preprocessed_data["Amplitudes"]
)
preprocessed_data["Cp_prepro"] = (
    preprocessed_data["Cp_prepro"]
    / preprocessed_data["Cp_prepro"].abs().max(1, keepdim=True).values
)

# Plot preprocessing
traces_line = []
traces_scatter = []
for i in range(preprocessed_data["Cp_prepro"].shape[0]):
    x = np.arange(0, len(preprocessed_data["Cp_prepro"][i]))
    y = preprocessed_data["Cp_prepro"][i].cpu().numpy()

    traces_line.append(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="royalblue"),
            opacity=0.5,
            legendgroup="Preprocessed",
            showlegend=(i == 0),
            name="Preprocessed" if i == 0 else None,
        )
    )
    traces_scatter.append(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=5, color="royalblue"),
            opacity=0.5,
            legendgroup="Preprocessed",
            showlegend=(i == 0),
            name="Preprocessed" if i == 0 else None,
        )
    )

traces_line.append(
    go.Scatter(
        x=np.arange(0, len(single_data["Cp"][0])),
        y=single_data["Cp"][0].cpu().numpy(),
        mode="lines",
        line=dict(color="red"),
        name="Original",
    )
)
traces_scatter.append(
    go.Scatter(
        x=np.arange(0, len(single_data["Cp"][0])),
        y=single_data["Cp"][0].cpu().numpy(),
        mode="markers",
        marker=dict(size=5, color="red"),
        name="Original",
    )
)

fig = go.Figure(data=traces_line)
fig.update_layout(
    title="Preprocessed Data",
    xaxis_title="Station Index",
    yaxis_title="Cp Preprocessed",
    showlegend=True,
)
fig.write_image(
    save_dir / "preprocessed_data.png",
    engine="kaleido",
    width=800,
    height=600,
    scale=2.0,
)
myshow(fig=fig)

fig = go.Figure(data=traces_scatter)
fig.update_layout(
    title="Preprocessed Data",
    xaxis_title="Station Index",
    yaxis_title="Cp Preprocessed",
    showlegend=True,
)
fig.write_image(
    save_dir / "preprocessed_data_scatter.png",
    engine="kaleido",
    width=800,
    height=600,
    scale=2.0,
)
myshow(fig=fig)

traces_amplitude_line = []
traces_amplitude_scatter = []
for i in range(preprocessed_data["Amplitudes"].shape[0]):
    x = np.arange(0, len(preprocessed_data["Amplitudes"][i]))
    y = preprocessed_data["Amplitudes"][i].cpu().numpy()

    traces_amplitude_line.append(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="royalblue"),
            opacity=0.5,
            legendgroup="Preprocessed",
            showlegend=(i == 0),
            name="Preprocessed" if i == 0 else None,
        )
    )
    traces_amplitude_scatter.append(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10, color="#9370DB"),
            opacity=0.5,
            legendgroup="Preprocessed",
            showlegend=(i == 0),
            name="Preprocessed" if i == 0 else None,
        )
    )


traces_amplitude_line.append(
    go.Scatter(
        x=np.arange(0, len(single_data["Amplitudes"][0])),
        y=single_data["Amplitudes"][0].cpu().numpy(),
        mode="lines",
        line=dict(color="red"),
        name="Original",
    )
)
traces_amplitude_scatter.append(
    go.Scatter(
        x=np.arange(0, len(single_data["Amplitudes"][0])),
        y=single_data["Amplitudes"][0].cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Original",
    )
)

fig = go.Figure(data=traces_amplitude_line)
fig.update_layout(
    xaxis_title="Station Index",
    yaxis_title="Amplitudes",
    showlegend=True,
)

fig.write_image(
    save_dir / "preprocessed_amplitudes_data.png",
    engine="kaleido",
    width=800,
    height=600,
    scale=2.0,
)
myshow(fig=fig)

fig = go.Figure(data=traces_amplitude_scatter)
fig.update_layout(
    xaxis_title="Station Index",
    yaxis_title="Amplitudes",
    showlegend=True,
    legend=dict(
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor="white",
        borderwidth=0,
        font=dict(size=18),
    ),
    margin=dict(l=5, r=30, t=30, b=5),
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(
        title_font=dict(size=24),
        tickfont=dict(size=20),
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.2)",
    ),
    yaxis=dict(
        title_font=dict(size=24),
        tickfont=dict(size=20),
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.2)",
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        zerolinewidth=1,
    ),
)

fig.write_image(
    save_dir / "preprocessed_amplitudes_data_scatter.png",
    engine="kaleido",
    width=1400,
    height=600,
    scale=2.0,
)
myshow(fig=fig)

# Line plot
plt.figure(figsize=(8, 6), dpi=160)

# Preprocessed data - lines
for i in range(preprocessed_data["Amplitudes"].shape[0]):
    x = np.arange(0, len(preprocessed_data["Amplitudes"][i]))
    y = preprocessed_data["Amplitudes"][i].cpu().numpy()
    plt.plot(
        x, y, color="royalblue", alpha=0.5, label="Preprocessed" if i == 0 else None
    )

# Original data - line
x_orig = np.arange(0, len(single_data["Amplitudes"][0]))
y_orig = single_data["Amplitudes"][0].cpu().numpy()
plt.plot(x_orig, y_orig, color="red", label="Original")

plt.xlabel("Station Index", fontsize=18)
plt.ylabel("Amplitudes", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(save_dir / "preprocessed_amplitudes_data.png", dpi=300, bbox_inches="tight")
myshow()


# Scatter plot
plt.figure(figsize=(14, 6), dpi=160)

# Preprocessed data - scatter
for i in range(preprocessed_data["Amplitudes"].shape[0]):
    x = np.arange(0, len(preprocessed_data["Amplitudes"][i]))
    y = preprocessed_data["Amplitudes"][i].cpu().numpy()
    plt.scatter(
        x, y, s=50, color="#9370DB", alpha=0.5, label="Preprocessed" if i == 0 else None
    )

# Original data - scatter
plt.scatter(x_orig, y_orig, s=50, color="red", label="Original")

plt.xlabel("Station Index", fontsize=18)
plt.ylabel("Amplitudes", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which="both", linestyle="--", color="gray", alpha=0.3)

plt.legend(
    loc="upper left", fontsize=16, frameon=True, facecolor="white", edgecolor="none"
)

plt.tight_layout()
plt.savefig(
    save_dir / "preprocessed_amplitudes_data_scatter.png", dpi=300, bbox_inches="tight"
)
myshow()

# %%

# This cell generates synthetic seismic data for a fixed focal mechanism
# but with hypocenters distributed along a diagonal line in the X, Y, Z space.
# It then calculates the Euclidean distance of each hypocenter from the first
# point on this diagonal. The primary goal is to visualize how the P-wave
# radiation pattern (Cp values) changes as the hypocenter moves along this
# diagonal path, using both line and scatter plots where the color of each
# curve/point represents its distance from the starting point. Plots are saved as PNG files.

xyz_stack = torch.stack(
    [
        torch.linspace(0, 2.0, 21),
        torch.linspace(0, 2.0, 21),
        torch.linspace(2, 3.0, 21),
    ],
    dim=1,
)

# Compute Euclidean distance from the first point
distances = torch.cdist(
    xyz_stack,
    xyz_stack[0].unsqueeze(0),
    p=2,
).squeeze(0)

fixed_sdr = FocalOnlineDataset(
    xyz_boundary=[-2.0, 2.0, -2.0, 2.0, 2.0, 3.0],
    sdr_boundary=[0, 0.0, 45.0, 45.0, 0, 0],
    xyz_steps=[0.1, 0.1, 0.05],
    sdr_steps=[2.5, 2.5, 2.5],
    exclude_strike=[360],
    exclude_dip=[0, 90],
    exclude_rake=[-180, 180],
    velmod_file="velmod_campi_flegrei.dat",
    stations_file="stations_campi_flegrei.dat",
    input_dir=input_dir,
    batch_size=int(2**11),
    generator=torch.Generator(device=device),
    device=device,
    xyz_translation_step_percs=None,
    xyz_points=xyz_stack,
)
data = fixed_sdr.gen_data()


def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Restituisce una nuova cmap che usa solo la porzione [minval, maxval]
    del colormap originale.
    """
    orig = plt.get_cmap(cmap)
    new_colors = orig(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap}_{minval:.2f}_{maxval:.2f}", new_colors
    )


cmap = truncate_cmap("plasma", minval=0.0, maxval=0.9)
norm = Normalize(vmin=distances.min(), vmax=distances.max())

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(data["Cp"].shape[0]):
    x = np.arange(0, len(data["Cp"][i]))
    y = data["Cp"][i].cpu().numpy()
    di = distances[i]
    color = cmap(norm(di))

    plt.plot(
        x,
        y,
        color=color,
        alpha=0.8,
    )

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Distanza")

plt.title(f"Hypocenter Data diagonally")
plt.xlabel("Station Index")
plt.ylabel("Cp Hypocenter")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(save_dir / f"hypocenter_data_diagonally.png", dpi=300)
myshow()

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(data["Cp"].shape[0]):
    x = np.arange(0, len(data["Cp"][i]))
    y = data["Cp"][i].cpu().numpy()
    di = distances[i]
    color = cmap(norm(di))

    ax.scatter(x, y, color=color, alpha=0.7)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Distance (km)", fontsize=18)
cbar.ax.tick_params(labelsize=16)
ax.set_xlabel("Station Index", fontsize=18)
ax.set_ylabel("Cp", fontsize=18)
ax.tick_params(axis="both", labelsize=16)
ax.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(save_dir / "hypocenter_data_scatter_diagonally.png", dpi=300)
myshow()


# %%

# This cell iterates through a list of different spatial boundaries (xyz_boundary)
# to generate synthetic seismic data. For each defined boundary, it creates data
# with a fixed focal mechanism but varying hypocenter locations within that specific
# spatial range. It then visualizes the P-wave radiation patterns (Cp values)
# for all generated hypocenters within each boundary using both line and scatter
# plots. Each plot is saved as a PNG file, with the filename reflecting the
# corresponding XYZ boundary.

lista_xyz = [
    [-0.1, 0.1, -0.1, 0.1, 2.4, 2.6],
    [-0.25, 0.25, -0.25, 0.25, 2.3, 2.7],
    [-0.5, 0.5, -0.5, 0.5, 2.2, 2.8],
    [-1.0, 1.0, -1.0, 1.0, 2.1, 2.9],
    [-2.0, 2.0, -2.0, 2.0, 2.0, 3.0],
]
for xyz in tqdm(lista_xyz):
    fixed_sdr = FocalOnlineDataset(
        xyz_boundary=xyz,
        sdr_boundary=[0, 0.0, 45.0, 45.0, 0, 0],
        xyz_steps=[0.1, 0.1, 0.05],
        sdr_steps=[2.5, 2.5, 2.5],
        exclude_strike=[360],
        exclude_dip=[0, 90],
        exclude_rake=[-180, 180],
        velmod_file="velmod_campi_flegrei.dat",
        stations_file="stations_campi_flegrei.dat",
        input_dir=input_dir,
        batch_size=int(2**11),
        generator=torch.Generator(device=device),
        device=device,
        xyz_translation_step_percs=None,
    )

    data = fixed_sdr.gen_data()

    angle_ref = [0.0, 45.0, 0.0]

    plt.figure(figsize=(10, 8))
    for i in range(data["Cp"].shape[0]):
        x = np.arange(0, len(data["Cp"][i]))
        y = data["Cp"][i].cpu().numpy()

        label = "Hypocenter" if i == 0 else None
        plt.plot(x, y, color="royalblue", alpha=0.5, label=label)

    plt.title(f"Hypocenter Data {xyz}")
    plt.xlabel("Station Index")
    plt.ylabel("Cp Hypocenter")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / f"hypocenter_data_{xyz}.png", dpi=300)
    myshow()

    plt.figure(figsize=(10, 8))
    for i in range(data["Cp"].shape[0]):
        x = np.arange(0, len(data["Cp"][i]))
        y = data["Cp"][i].cpu().numpy()

        label = "Hypocenter" if i == 0 else None
        plt.scatter(x, y, s=5, color="royalblue", alpha=0.5, label=label)

    plt.title(f"Hypocenter Data {xyz}")
    plt.xlabel("Station Index")
    plt.ylabel("Cp Hypocenter")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_dir / f"hypocenter_data_scatter_{xyz}.png", dpi=300)
    myshow()

# %%
