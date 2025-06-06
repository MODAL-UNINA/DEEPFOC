#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%
# IMPORTS

import os
import yaml
import torch
import pickle
import random
import shutil
import importlib
import numpy as np
import mplstereonet
import pandas as pd
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from fastcore.all import dict2obj, L
from tqdm.auto import trange, tqdm
from shapely.geometry import Point
from IPython.display import display
from strec.kagan import get_kagan_angle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Custom modules for losses, synthetic data generation, plotting, dataset creation, etc.
from utils.loss import KaganAngle
from utils.synthetic import radpar
from utils.seismology_utils import plot_focal_mechanisms_pt, compute_discrepancy_error
from utils.online_dataset import FocalOnlineDataset
from utils.utils import create_test_dataset, compute_figure_rows_cols

# FPfit functions for focal mechanism inversion
from fpyfit.fpyfit import read_velmod_data, read_stations_data, check_and_gclc, fpyfit

# %%
# SETUP DIRECTORIES AND PARAMETERS

scripts_dir = Path(os.getcwd()).resolve()
root_dir = scripts_dir.parent

# Folder name where trained model is saved and GPU device id (-1 for CPU suggested)
id_device = -1

# Define paths for saving results, input and output directories
save_folders = ["zone1", "zone2", "zone3"]  # Example folder names, adjust as needed

for save_folder in save_folders:
    save_dir = root_dir / "results" / save_folder
    data_dir = root_dir / "data" 
    input_dir = data_dir / "INPUT"
    model_dir = scripts_dir / "MODELS" / save_folder
    if save_dir.exists():
        shutil.rmtree(save_dir)  # Remove existing results folder if it exists

    # Load arguments from YAML file and convert to object for attribute access
    with open(model_dir / "args.yml", "r") as f:
        args = yaml.safe_load(f)
        args = dict2obj(args)

    for key, value in args.items():
        if isinstance(value, L):
            setattr(args, key, list(value))

    # Ensure the FPfit folder name is not too long
    if len(save_folder) > 30:
        fpfit_folder = list(save_folder.lower().replace("_", ""))[:30]
    else:
        fpfit_folder = list(save_folder.lower().replace("_", ""))
    random.shuffle(fpfit_folder)
    fpfit_folder = "".join(fpfit_folder)

    # Set the GPU device for CUDA operations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id_device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    # # # # -----------------------------------------------------------------------------
    # PREPARE STATION DATA

    # Read velocity model data (input: file path and numpy dtype; output: reference latitude, reference longitude, velocity, depth)
    latref, lonref, v, d = read_velmod_data(
        input_dir / args.velmod_file, np.dtype("float64")
    )

    # Read station data (input: file path and numpy dtype; output: station names, latitudes, longitudes, elevations)
    staz, lats, lons, elev = read_stations_data(
        input_dir / args.stations_file, np.dtype("float64")
    )

    # Compute corrected station coordinates (output: y and x positions for stations)
    ystaz, xstaz = check_and_gclc(latref, lonref, lats, lons)

    # Build a DataFrame with station information for easy access
    df_stations = pd.DataFrame(
        {
            "staz": staz,
            "xstaz": xstaz,
            "ystaz": ystaz,
            "lats": lats,
            "lons": lons,
            "elev": elev,
        }
    )

    xstaz_tensor = torch.tensor(xstaz)
    ystaz_tensor = torch.tensor(ystaz)

    # Define column names for various purposes
    xyz_cols = ["x", "y", "z"]
    strikediprake_cols = ["strike", "dip", "rake"]
    cp_cols = [f"cp_{i}" for i in df_stations.staz.values]
    pol_cols = [f"pol_{i}" for i in df_stations.staz.values]
    amplitude_cols = [f"amplitude_{i}" for i in df_stations.staz.values]
    az_cols = [f"az_{i}" for i in df_stations.staz.values]
    ih_cols = [f"ih_{i}" for i in df_stations.staz.values]
    cp_columns = [f"cp_{i}" for i in df_stations.staz.values]

    # Load real test datasets (Real and Synthetic) and fill missing values with 0
    df_real_test = pd.read_csv(data_dir / "REAL_TEST_CAMPI_FLEGREI/test_real.csv")

    # Check if a specific polygon zone is provided in the arguments
    if hasattr(args, "polygon_zone") and args.polygon_zone is not None:
        # Read the GeoJSON file containing geographical zones
        zones_gpd = gpd.read_file(input_dir / "Campi_Flegrei_zones.geojson")
        # Set the 'Zone' column as the index for easy lookup
        zones_gpd = zones_gpd.set_index("Zone")
        # Select the geometry (polygon) corresponding to the specified polygon_zone
        polygon_zone = zones_gpd.loc[args.polygon_zone, "geometry"]
        # Filter the Test DataFrame to include only points (events) that fall within the selected polygon zone
        df_real_test = df_real_test[
            df_real_test.apply(
                lambda row: Point(row["x"], row["y"]).within(polygon_zone), axis=1
            )
        ]
    else:
        # If no polygon zone is specified, filter the Test DataFrame based on XYZ boundaries
        df_real_test = df_real_test[
            (df_real_test["x"] >= args.xyz_boundary[0])  # Filter by minimum X coordinate
            & (df_real_test["x"] <= args.xyz_boundary[1])  # Filter by maximum X coordinate
            & (df_real_test["y"] >= args.xyz_boundary[2])  # Filter by minimum Y coordinate
            & (df_real_test["y"] <= args.xyz_boundary[3])  # Filter by maximum Y coordinate
            & (df_real_test["z"] >= args.xyz_boundary[4])  # Filter by minimum Z coordinate
            & (df_real_test["z"] <= args.xyz_boundary[5])  # Filter by maximum Z coordinate
        ]
        # Set polygon_zone to None as no geographical polygon was used for filtering
        polygon_zone = None

    # Reset the index of the filtered DataFrame and drop the old index
    df_real_test.reset_index(drop=True, inplace=True)
    # Fill any NaN values in columns specified by 'cp_cols' with 0
    df_real_test.loc[:, cp_cols] = df_real_test.loc[:, cp_cols].fillna(0)
    # Fill any NaN values in columns specified by 'pol_cols' with 0
    df_real_test.loc[:, pol_cols] = df_real_test.loc[:, pol_cols].fillna(0)
    # Fill any NaN values in columns specified by 'amplitude_cols' with -1
    df_real_test.loc[:, amplitude_cols] = df_real_test.loc[:, amplitude_cols].fillna(-1)
    # Extract event names from the 'name' column of the real test dataset
    name_event = df_real_test["name"].values

    # # # # -----------------------------------------------------------------------------
    # DYNAMIC MODULE IMPORT & MODEL INITIALIZATION

    # Construct the module name dynamically based on the output folder name
    module_name = f"{model_dir.parent.name}.{model_dir.name}.model"

    # Import the module dynamically (the module contains the model definition)
    module = importlib.import_module(module_name)

    # Access the CustomModel class from the imported module using the model name specified in args
    CustomModel = getattr(module, args.model_name)

    # Initialize the model
    # Input Attributes:
    #   - xyz_boundary: list specifying the spatial boundary for the model
    #   - scaling_range: list specifying the scaling parameters
    # Output:
    #   - model: an instance of CustomModel configured and moved to the GPU with the desired float type
    model = (
        CustomModel(
            n_stations=len(staz),
            xyz_boundary=list(args.xyz_boundary),
            scaling_range=list(args.scaled_parameters),
        )
        .to(device)
        .to(getattr(torch, args.float_type))
    )
    # Load the best model parameters from file and set the model to evaluation mode
    model.load_parameters_correctly(model_dir / "best_model.pth")
    model.eval()

    # Define the loss function for focal mechanism predictions (Kagan loss)
    KaganLoss = KaganAngle(reduction="mean", normalize=False)


    # Prepare test datasets for the model.
    # The create_test_dataset function takes:
    #   - a DataFrame with event data,
    #   - station names, and
    #   - the desired float type.
    # It returns a dictionary with properly formatted input data for the model.
    real_complete = {
        "Real": create_test_dataset(
            df_real_test, df_stations.staz.values, args.float_type, device=device
        )
    }

    # # # # -----------------------------------------------------------------------------
    save_dir_train_metrics = save_dir / "Analysis" / "Train_Metrics"
    # Create the directory if it does not already exist, and create any necessary parent directories.
    save_dir_train_metrics.mkdir(exist_ok=True, parents=True)

    # Open the 'metrics.pkl' file located in 'save_dir' in binary read mode ('rb').
    # This file is expected to contain a dictionary of training and validation metrics.
    with open(model_dir / "metrics.pkl", "rb") as f:
        # Load the metrics data from the pickle file.
        metrics = pickle.load(f)

    # Create a new figure and a single subplot (Axes object) for plotting.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the training loss over epochs.
    ax.plot(metrics["Train"]["Loss"], label="Train Loss")
    # Plot the validation loss over epochs.
    ax.plot(metrics["Validation"]["Loss"], label="Validation Loss")
    # Add vertical dashed red lines at the epochs corresponding to the two lowest validation losses.
    # np.argpartition finds the indices that would partition the array,
    # so [:2] gets the indices of the two smallest values.
    ax.vlines(
        np.argpartition(metrics["Validation"]["Loss"], 2)[:2],
        ymin=np.min(
            [np.min(metrics["Validation"]["Loss"]), np.min(metrics["Train"]["Loss"])]
        ),  # Minimum y-value for the vertical lines.
        ymax=np.max(
            metrics["Validation"]["Loss"]
        ),  # Maximum y-value for the vertical lines.
        color="red",  # Color of the lines.
        linestyle="--",  # Style of the lines (dashed).
    )
    # Set the label for the x-axis.
    ax.set_xlabel("Epoch")
    # Set the label for the y-axis.
    ax.set_ylabel("Loss")
    # Display the legend to identify the plotted lines.
    ax.legend()
    plt.grid(True)
    # Save the figure as a PNG image in the 'save_dir_train_metrics' directory.
    # 'dpi=200' sets the resolution
    fig.savefig(save_dir_train_metrics / "Loss.png", dpi=200, bbox_inches="tight")
    # Close the figure to free up memory.
    plt.close()

    # Create another new figure and a single subplot for plotting the same data on a logarithmic scale.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the training loss.
    ax.plot(metrics["Train"]["Loss"], label="Train Loss")
    # Plot the validation loss.
    ax.plot(metrics["Validation"]["Loss"], label="Validation Loss")
    # Add vertical dashed red lines at the epochs corresponding to the two lowest validation losses.
    ax.vlines(
        np.argpartition(metrics["Validation"]["Loss"], 2)[:2],
        ymin=np.min(
            [np.min(metrics["Validation"]["Loss"]), np.min(metrics["Train"]["Loss"])]
        ),
        ymax=np.max(metrics["Validation"]["Loss"]),
        color="red",
        linestyle="--",
    )
    # Set the label for the x-axis.
    ax.set_xlabel("Epoch")
    # Set the label for the y-axis.
    ax.set_ylabel("Loss")
    # Set the y-axis to a logarithmic scale.
    ax.set_yscale("log")
    # Display the legend.
    ax.legend()
    plt.grid(True)
    # Save this figure as a PNG image, named 'Loss_log.png'.
    fig.savefig(save_dir_train_metrics / "Loss_log.png", dpi=200, bbox_inches="tight")
    # Close the figure.
    plt.close()


    # # # # -----------------------------------------------------------------------------
    # MODEL PERFORMANCE EVALUATION

    with torch.no_grad():
        # Create a loss function that returns individual loss values (no reduction)
        KaganLoss_nored = KaganAngle(reduction=None, normalize=False)

        # Iterate over each test dataset in real_complete
        for key, batch in real_complete.items():
            # Add station positions to the batch (input: xstaz_tensor and ystaz_tensor; output: staz_pos field added to batch)
            df_model_predictions = pd.DataFrame(
                columns=["Strike", "Dip", "Rake"],
                index=list(name_event),
                dtype=np.float64,
            )
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            batch["staz_pos"] = torch.stack(
                [
                    xstaz_tensor.repeat(len(batch["XYZ"]), 1),
                    ystaz_tensor.repeat(len(batch["XYZ"]), 1),
                ],
                dim=2,
            ).to(device)
            outputs = model.predict_angles(model(batch))
            df_model_predictions.loc[:, ["Strike", "Dip", "Rake"]] = np.round(
                outputs.cpu().numpy(), 2
            )
            df_model_predictions.to_csv(save_dir / f"model_predictions_{key}.csv")
            df_model_predictions.to_excel(save_dir / f"model_predictions_{key}.xlsx")

    # # # # -----------------------------------------------------------------------------
    # FPfit RESULTS COMPUTATION
    #
    # Inputs:
    #   df_real_test     : pd.DataFrame
    #                      Test events with CP columns and (possibly) polarity columns.
    #   df_stations      : pd.DataFrame
    #                      Station metadata with 'staz', 'xstaz', 'ystaz', etc.
    #   pol_cols, cp_cols: List[str]
    #                      Column names for polarities and CP amplitudes.
    #   latref, lonref   : float
    #                      Reference latitude and longitude for coordinate conversion.
    #   fpyfit           : Callable
    #                      Function to run FPfit inversion on a DataFrame of events.
    #   save_dir         : pathlib.Path
    #                      Directory to save the FPfit results CSV.
    #
    # Output:
    #   df_fpfit_result  : pd.DataFrame
    #                      Flattened FPfit solutions with columns ['id','strike','dip','rake'].
    fpfit_test_results = {}

    # If polarity columns are absent, derive them from sign of CP amplitudes
    if np.all([pol not in df_real_test.columns for pol in pol_cols]):
        df_real_test.loc[:, pol_cols] = np.sign(df_real_test.loc[:, cp_cols].values)

    # Run FPfit for the real test dataset
    # fpyfit function inputs:
    #   - DataFrame with event data,
    #   - DataFrame with station data,
    #   - Reference lat/lon,
    #   - file_name (for output),
    #   - max_threads (for parallel processing)
    # Output:
    #   - A dictionary with FPfit solutions per event.
    fpfit_test_results["Real"] = fpyfit(
        df_real_test,
        df_stations,
        latref_lonref=[latref, lonref],
        file_name=save_dir / f"{fpfit_folder}.prt",
        max_threads=1,
    )

    # Flatten the dictionary of lists of solutions into a DataFrame
    rows = []
    for event_id, solutions in fpfit_test_results["Real"].items():
        if solutions is None:
            continue
        for solution in solutions:
            row = {"id": event_id}
            row.update(solution)
            rows.append(row)

    df_fpfit_result = pd.DataFrame(rows)
    df_fpfit_result.to_csv(save_dir / "fpfit_results.csv", index=False)

    # # # # -----------------------------------------------------------------------------
    # BUILD FINAL EVENT TABLE
    #
    # Combines event metadata, model predictions, and FPfit solutions into one table.
    #
    # Inputs:
    #   df_real_test       : pd.DataFrame  (must contain 'name','latitude','longitude','z')
    #   df_model_predictions: pd.DataFrame (index=name_event, columns=['Strike','Dip','Rake'])
    #   df_fpfit_result     : pd.DataFrame (columns=['id','strike','dip','rake'])
    #
    # Output:
    #   df_events : pd.DataFrame with columns
    #       ['name','lat','lon','z','strike','dip','rake','fpfit_strike','fpfit_dip','fpfit_rake']
    df_events: pd.DataFrame = pd.DataFrame(
        columns=[
            "name",
            "lat",
            "lon",
            "z",
            "strike",
            "dip",
            "rake",
            "fpfit_strike",
            "fpfit_dip",
            "fpfit_rake",
        ],
        index=df_real_test.index,
    )
    for i, row in tqdm(
        df_real_test.iterrows(), total=len(df_real_test), desc="Creating df_eventi"
    ):
        # Basic event metadata
        df_events.loc[i, "name"] = row["name"]
        df_events.loc[i, "lat"] = row["latitude"]
        df_events.loc[i, "lon"] = row["longitude"]
        df_events.loc[i, "z"] = row["z"]
        # Model predictions
        df_events.loc[i, "strike"] = df_model_predictions.loc[row["name"], "Strike"]
        df_events.loc[i, "dip"] = df_model_predictions.loc[row["name"], "Dip"]
        df_events.loc[i, "rake"] = df_model_predictions.loc[row["name"], "Rake"]

        # FPfit solutions (mark with "*" if multiple solutions exist)
        if row["name"] in df_fpfit_result["id"].values:
            fpfit_row = df_fpfit_result[df_fpfit_result["id"] == row["name"]]
            if len(fpfit_row) > 1:
                # Indicate ambiguity by appending "*"
                df_events.loc[i, "fpfit_strike"] = str(fpfit_row["strike"].values[0]) + "*"
                df_events.loc[i, "fpfit_dip"] = str(fpfit_row["dip"].values[0]) + "*"
                df_events.loc[i, "fpfit_rake"] = str(fpfit_row["rake"].values[0]) + "*"
            else:
                df_events.loc[i, "fpfit_strike"] = fpfit_row["strike"].values[0]
                df_events.loc[i, "fpfit_dip"] = fpfit_row["dip"].values[0]
                df_events.loc[i, "fpfit_rake"] = fpfit_row["rake"].values[0]

        else:
            # No solution found
            df_events.loc[i, "fpfit_strike"] = np.nan
            df_events.loc[i, "fpfit_dip"] = np.nan
            df_events.loc[i, "fpfit_rake"] = np.nan

    # Save the combined event table
    df_events.to_csv(save_dir / "results_deepfoc_and_fpfit.csv", index=False)

    # # # # -----------------------------------------------------------------------------

    # Get model predictions for the "Real" dataset
    with torch.no_grad():
        # Use the model to predict angles on the deep-copied "Real" dataset
        SDR_model_test = model.predict_angles(
            model(deepcopy(real_complete["Real"]))
        ).detach()

    # Make a copy of the "Real" dataset for CP calculation
    real_complete_Real = deepcopy(real_complete["Real"])
    # Preprocess amplitudes: set negative values to zero, detach, move to CPU, convert to NumPy,
    # then apply polarities
    real_complete_Real_cp = (
        torch.where(
            real_complete_Real["Amplitudes"] < 0,  # Identify negative amplitudes
            0,  # Replace negatives with zero
            real_complete_Real["Amplitudes"],  # Keep positive amplitudes
        )
        .detach()  # Remove from computation graph
        .cpu()  # Move tensor to CPU memory
        .numpy()  # Convert to NumPy array
        * real_complete_Real["Polarities"].detach().cpu().numpy()  # Multiply by polarities
    )

    # # # # -----------------------------------------------------------------------------
    # Set up CP comparisons directory, compute CP values for each event, and plot comparisons
    # Define and create directory for saving CP comparison plots
    save_dir_cp_comparisons = save_dir / "Plots" / "Cp_comparisons"
    save_dir_cp_comparisons.mkdir(exist_ok=True, parents=True)

    # Compute RMSE between Real, DeepFoc, and FPFIT CP values, summarize in DataFrame, and plot
    # Initialize DataFrame for RMSE results, indexed by event name with comparison columns
    df_rmse = pd.DataFrame(
        index=df_real_test["name"].values,
        columns=["Real vs DeepFoc", "Real vs FPFIT", "DeepFoc vs FPFIT"],
    )

    # Loop over each event to calculate RMSE metrics and generate comparison plots
    for i_dft, row in tqdm(
        df_real_test.iterrows(), total=len(df_real_test), desc="Plotting"
    ):
        # Extract CP values for the event and mask out zeros
        row_vals = df_real_test.loc[i_dft, cp_cols]
        mask_valid_cp = row_vals != 0.0
        valid_cp_cols = row_vals[mask_valid_cp].index.tolist()
        # Map CP column names to corresponding azimuth and incidence columns
        valid_az_cols = [col.replace("cp_", "az_") for col in valid_cp_cols]
        valid_ih_cols = [col.replace("cp_", "ih_") for col in valid_cp_cols]

        # Retrieve and sort azimuth, incidence, and CP arrays by azimuth
        az_vals = df_real_test.loc[i_dft, valid_az_cols].values.astype(np.float64)
        ih_vals = df_real_test.loc[i_dft, valid_ih_cols].values.astype(np.float64)
        cp_vals = df_real_test.loc[i_dft, valid_cp_cols].values.astype(np.float64)

        order = np.argsort(az_vals)
        az_vals_sorted = az_vals[order]
        ih_vals_sorted = ih_vals[order]
        cp_vals_sorted = cp_vals[order]

        # Compute model CP via radpar and store with real CP in cp_results list
        cp_results = []
        cp_results.append(cp_vals_sorted)  # Real CP
        cp_results.append(
            radpar(
                torch.tensor(az_vals_sorted).to(device),
                SDR_model_test[i_dft].to(device),
                torch.tensor(ih_vals_sorted).to(device),
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        cp_real = cp_results[0]
        cp_model = cp_results[1]

        # Retrieve FPFIT focal mechanism and compute its CP if available
        sdr_fpfit = df_fpfit_result[df_fpfit_result["id"] == row["name"]][
            ["strike", "dip", "rake"]
        ].values
        if len(sdr_fpfit) > 0:
            for sdr in sdr_fpfit:
                cp_results.append(
                    radpar(
                        torch.tensor(az_vals_sorted).to(device),
                        torch.tensor(sdr).to(device).squeeze(),
                        torch.tensor(ih_vals_sorted).to(device),
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

        cp_fpfit = cp_results[2:]  # List of FPFIT CP arrays

        # Compute RMSE between model and real CP
        rmse_model = np.sqrt(np.mean((cp_model - cp_real) ** 2))

        # Compute RMSE arrays for FPFIT vs real and FPFIT vs model
        rmse_fpfit = []
        rmse_fpfit_model = []
        for cp in cp_fpfit:
            rmse_fpfit.append(np.sqrt(np.mean((cp - cp_real) ** 2)))
            rmse_fpfit_model.append(np.sqrt(np.mean((cp - cp_model) ** 2)))
        # Store mean RMSE values in the DataFrame
        df_rmse.loc[row["name"], "Real vs DeepFoc"] = rmse_model
        df_rmse.loc[row["name"], "Real vs FPFIT"] = np.mean(rmse_fpfit)
        df_rmse.loc[row["name"], "DeepFoc vs FPFIT"] = np.mean(rmse_fpfit_model)

        # Plot real, model, and FPFIT CP curves with RMSE annotations
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes = np.atleast_1d(axes)
        # Plot normalized real CP curve
        cp = cp_results[0] / np.max(np.abs(cp_results[0]))
        axes.flatten()[0].plot(cp, label="Real", color="black", marker="o")
        axes.flatten()[0].set_xticks(range(len(valid_az_cols)))
        axes.flatten()[0].set_ylim(-1.2, 1.2)
        axes.flatten()[0].set_yticks(np.arange(-1, 1.1, 0.4))

        # Label x-ticks with station indices
        axes.flatten()[0].set_xticklabels(
            [col.replace("az_", "") for col in np.array(valid_az_cols)[order]], rotation=45
        )

        # Plot normalized model CP with RMSE in legend
        cp = cp_results[1] / np.max(np.abs(cp_results[1]))
        axes.flatten()[0].plot(
            cp, label=f"DeepFoc - RMSE {rmse_model.round(3)}", color="red", linestyle="--"
        )

        # Plot each FPFIT CP curve with its RMSE
        colors = sns.color_palette("Set2", len(cp_results[2:]))
        for i_rcr, cp in enumerate(cp_results[2:]):
            cp = cp / np.max(np.abs(cp))
            axes.flatten()[0].plot(
                cp,
                label=f"FPFIT_{i_rcr} - RMSE {rmse_fpfit[i_rcr].round(3)}",
                color=colors[i_rcr],
            )
        axes.flatten()[0].legend(fontsize=16)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        fig.subplots_adjust(right=0.75)

        # Adjust tick label size and add title
        axes.flatten()[0].tick_params(axis="both", labelsize=16)
        fig.suptitle(f"{row['name']}", fontsize=20)
        fig.tight_layout()
        fig.savefig(
            save_dir_cp_comparisons / f"real_model_fpfit_{row['name']}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # After looping, compute mean and std deviation rows for RMSE results
    rows = df_rmse.index
    df_rmse.loc["Mean"] = df_rmse.loc[rows].mean(0)
    df_rmse.loc["Std"] = df_rmse.loc[rows].std(0)

    # Save RMSE results to CSV for external analysis
    df_rmse.to_csv(save_dir / "RMSE_results.csv", index=True)

    # # # # -----------------------------------------------------------------------------

    # This cell iterates through a dataset of real seismic events, processing each
    # event to compare predicted focal mechanisms (from a trained model) with
    # observed focal mechanisms and, when available, with results from the FPFIT
    # software. For each event, it generates and saves 'beachball' plots
    # (stereonet projections) that visually represent these focal mechanisms.
    # The plots include information on discrepancy errors, RMSE, and Kagan angles,
    # providing a comprehensive comparison of model performance against ground truth
    # and traditional methods. FPFIT results are plotted on one subplot,
    # while the model predictions and observed data are plotted on another.

    # Create directory for beachball plots
    (save_dir / "Plots" / "BeachBall").mkdir(exist_ok=True, parents=True)

    for key, batch in tqdm(real_complete.items(), desc="Plotting BeachBall"):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        batch["staz_pos"] = torch.stack(
            [
                xstaz_tensor.repeat(batch["Amplitudes"].shape[0], 1),
                ystaz_tensor.repeat(batch["Amplitudes"].shape[0], 1),
            ],
            dim=2,
        ).to(device)

        outputs = model(batch)
        SDR_pred = model.predict_angles(outputs).detach()
        targets = batch["SDR"].to(device)
        idx_not_nan = torch.all(~torch.isnan(targets), dim=1)
        kagan_loss = torch.empty(len(targets)).fill_(np.nan)
        kagan_loss[idx_not_nan] = KaganLoss(SDR_pred[idx_not_nan], targets[idx_not_nan])[
            "Kagan"
        ]

        for ii in range(len(SDR_pred)):

            az = real_complete[key]["az"].detach().cpu().numpy()[ii]
            ih = real_complete[key]["ih"].detach().cpu().numpy()[ii]
            pol = real_complete[key]["Polarities"].detach().cpu().numpy()[ii]

            rmse_ii = df_rmse.loc[
                df_rmse.index == name_event[ii], ["Real vs DeepFoc", "Real vs FPFIT"]
            ].values.squeeze()

            df_stat_obs = pd.DataFrame(
                data={
                    "station": df_stations.staz.values,
                    "azimuth": az,
                    "takeoff": ih,
                    "polarity": pol,
                    "amplitudes": torch.where(
                        real_complete[key]["Amplitudes"] < 0,
                        0,
                        real_complete[key]["Amplitudes"],
                    )
                    .detach()
                    .cpu()
                    .numpy()[ii],
                },
                columns=["station", "azimuth", "takeoff", "polarity", "amplitudes"],
            )
            df_stat_obs = df_stat_obs[
                (df_stat_obs["polarity"] != 0)
                & (df_stat_obs["azimuth"] != 0)
                & (df_stat_obs["takeoff"] != 0)
            ]

            y_hat = SDR_pred[ii].cpu()
            y = batch["SDR"].detach().cpu().numpy()[ii]
            have_observed = not np.isnan(y).any()

            if fpfit_test_results[key][name_event[ii]] is not None and have_observed:
                rows, cols = 1, 3
                i_count = 2
            elif fpfit_test_results[key][name_event[ii]] is not None or have_observed:
                rows, cols = 1, 2
                i_count = 1
            else:
                rows, cols = 1, 1

            fig, axes = mplstereonet.subplots(
                projection="equal_area",
                figsize=[8 * cols, 10 * rows],
                nrows=rows,
                ncols=cols,
            )
            axes = np.atleast_1d(axes)
            discrepancy = compute_discrepancy_error(
                y_hat.squeeze(),
                pol.squeeze(),
                az.squeeze(),
                ih.squeeze(),
            )
            fig, ax = plot_focal_mechanisms_pt(
                pd.DataFrame(
                    data={
                        "strike": [y_hat[0].item()],
                        "dip": [y_hat[1].item()],
                        "rake": [y_hat[2].item()],
                    }
                ),
                df_stat_obs,
                alpha=1,
                title=f"Focal Mechanism {key} DeepFoc \n {name_event[ii]}  \n Discrepancy: {discrepancy}\n RMSE {rmse_ii[0]:.3f} \n \n",
                ax=axes.flatten()[0],
                use_color=True,
            )

            if have_observed:
                discrepancy = compute_discrepancy_error(
                    sdr_np.squeeze(),
                    pol.squeeze(),
                    az.squeeze(),
                    ih.squeeze(),
                )
                fig, ax = plot_focal_mechanisms_pt(
                    pd.DataFrame(data={"strike": [y[0]], "dip": [y[1]], "rake": [y[2]]}),
                    df_stat_obs,
                    alpha=1,
                    title=f"Focal Mechanism {key} Observed \n {name_event[ii]}  \n Discrepancy: {discrepancy} \n \n",
                    use_color=True,
                    ax=axes.flatten()[1],
                )
                suptitle_str = f"Kagan Loss DeepFoc: {KaganLoss(model.predict_angles(outputs[ii].unsqueeze(0)), batch['SDR'][ii].unsqueeze(0))['Kagan'].item()}"
            else:
                suptitle_str = ""

            if fpfit_test_results[key][name_event[ii]] is not None:
                sdr = pd.DataFrame(
                    data=fpfit_test_results[key][name_event[ii]][0],
                    index=[0],
                )
                sdr = sdr.rename(
                    columns={
                        "strike": "strike",
                        "dip": "dip",
                        "rake": "rake",
                    }
                )
                sdr_np = np.array([sdr["strike"], sdr["dip"], sdr["rake"]])

                discrepancy = compute_discrepancy_error(
                    sdr_np.squeeze(),
                    pol.squeeze(),
                    az.squeeze(),
                    ih.squeeze(),
                )
                fig, ax = plot_focal_mechanisms_pt(
                    pd.DataFrame(data=fpfit_test_results[key][name_event[ii]]),
                    df_stat_obs,
                    alpha=1,
                    title=f"Focal Mechanism {key} FPfit \n {name_event[ii]} \n Discrepancy: {discrepancy}\n RMSE {rmse_ii[1]:.3f} \n \n",
                    use_color=True,
                    ax=axes.flatten()[i_count],
                )
                if have_observed:
                    suptitle_str += f"\nKagan Loss FPFIT: {[get_kagan_angle(sdr['strike'], sdr['dip'], sdr['rake'], *y) for sdr in fpfit_test_results[key][name_event[ii]]]}"
                else:
                    suptitle_str += f"\nKagan Loss FPFIT: {[get_kagan_angle(sdr['strike'], sdr['dip'], sdr['rake'], *y_hat) for sdr in fpfit_test_results[key][name_event[ii]]]}"

            fig.suptitle(suptitle_str)
            fig.savefig(
                save_dir / "Plots" / "BeachBall" / f"{key}_{name_event[ii]}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

    # # # # -----------------------------------------------------------------------------
    # JACKKNIFE TEST: NOISE TEST
    # This cell implements a jackknife test to evaluate the robustness of the model's
    # focal mechanism predictions to noise or missing data. It iterates through each
    # real seismic event, systematically removing data from one station at a time,
    # and then predicting the focal mechanism. It compares these jackknife solutions
    # against the solution obtained using all stations and, if available, FPFIT results.
    # The code generates and saves beachball plots for both DeepFoc and FPFIT,
    # highlighting the variations in focal mechanisms due to the removal of individual stations.
    # Finally, it combines these beachball plots with pre-existing Cp comparison plots,
    # providing a comprehensive view of the model's performance under noisy conditions.

    save_noise_test = save_dir / "Plots" / "Jackknife"
    save_noise_test.mkdir(exist_ok=True, parents=True)

    real_data = deepcopy(real_complete["Real"])

    for k in real_data.keys():
        if isinstance(real_data[k], torch.Tensor):
            real_data[k] = real_data[k].to(device)

    if "staz_pos" not in real_data.keys():
        real_data["staz_pos"] = torch.stack(
            [
                xstaz_tensor.repeat(real_data["Amplitudes"].shape[0], 1),
                ystaz_tensor.repeat(real_data["Amplitudes"].shape[0], 1),
            ],
            dim=2,
        ).to(device)
    else:
        real_data["staz_pos"] = real_data["staz_pos"].to(device)


    for idx_real_data in trange(len(real_data["Amplitudes"]), desc="Jacknife Noise Test"):
        real_data_i = {k: v[idx_real_data].unsqueeze(0) for k, v in real_data.items()}

        real_data_idx_staz = np.where(
            real_data_i["Polarities"][0].detach().cpu().numpy() != 0
        )[0]
        real_data_i_staz = np.array(staz)[real_data_idx_staz]
        rows, cols = compute_figure_rows_cols(len(real_data_i_staz))

        y_hat_complete = model.predict_angles(model(real_data_i)).detach().squeeze(0)
        df_jacknife = pd.DataFrame(
            columns=["strike", "dip", "rake", "kagan"], index=real_data_idx_staz
        )
        df_jacknife.loc["Complete"] = {
            "strike": y_hat_complete[0].item(),
            "dip": y_hat_complete[1].item(),
            "rake": y_hat_complete[2].item(),
            "kagan": KaganLoss(y_hat_complete, y_hat_complete)["Kagan"]
            .detach()
            .cpu()
            .item(),
        }
        complete_fpfit = fpyfit(
            df_real_test.iloc[[idx_real_data]],
            df_stations,
            latref_lonref=[latref, lonref],
            file_name=f"{fpfit_folder}.prt",
            max_threads=64,
            use_tqdm=False,
        )

        df_jacknife_fpfit = pd.DataFrame(columns=["strike", "dip", "rake", "kagan"])
        if complete_fpfit[name_event[idx_real_data]] is not None:
            for i, sol in enumerate(complete_fpfit[name_event[idx_real_data]]):
                strike = sol["strike"]
                dip = sol["dip"]
                rake = sol["rake"]

                df_jacknife_fpfit.loc[f"Complete_{i}"] = {
                    "strike": strike,
                    "dip": dip,
                    "rake": rake,
                    "kagan": get_kagan_angle(
                        strike,
                        dip,
                        rake,
                        y_hat_complete[0].item(),
                        y_hat_complete[1].item(),
                        y_hat_complete[2].item(),
                    ),
                }
        else:
            print(f"No FPFIT solution for {name_event[idx_real_data]}")

        for i_zero_staz, zero_staz in enumerate(real_data_idx_staz):
            real_data_i_temp = deepcopy(real_data_i)
            real_data_i_temp["Polarities"][0, zero_staz] = 0
            real_data_i_temp["Amplitudes"][0, zero_staz] = -1
            df_real_test_i_temp = deepcopy(df_real_test.iloc[[idx_real_data]])

            staz_name = real_data_i_staz[i_zero_staz]
            df_real_test_i_temp.loc[:, f"pol_{staz_name}"] = 0
            df_real_test_i_temp.loc[:, f"amplitudes_{staz_name}"] = -1

            jacknife_fpfit = fpyfit(
                df_real_test_i_temp,
                df_stations,
                latref_lonref=[latref, lonref],
                file_name=f"{fpfit_folder}.prt",
                max_threads=64,
                use_tqdm=False,
            )
            if jacknife_fpfit[name_event[idx_real_data]] is not None:
                for i, sol in enumerate(jacknife_fpfit[name_event[idx_real_data]]):
                    strike = sol["strike"]
                    dip = sol["dip"]
                    rake = sol["rake"]

                    df_jacknife_fpfit.loc[f"{zero_staz}_{i}"] = {
                        "strike": strike,
                        "dip": dip,
                        "rake": rake,
                        "kagan": get_kagan_angle(
                            strike,
                            dip,
                            rake,
                            y_hat_complete[0].item(),
                            y_hat_complete[1].item(),
                            y_hat_complete[2].item(),
                        ),
                    }

            outputs = model(real_data_i_temp)
            SDR_pred = model.predict_angles(outputs).detach()

            az = real_data_i_temp["az"].detach().cpu().numpy()[0]
            ih = real_data_i_temp["ih"].detach().cpu().numpy()[0]
            pol = real_data_i_temp["Polarities"].detach().cpu().numpy()[0]

            df_stat_obs = pd.DataFrame(
                data={
                    "station": df_stations.staz.values,
                    "azimuth": az,
                    "takeoff": ih,
                    "polarity": pol,
                    "amplitudes": torch.where(
                        real_data_i_temp["Amplitudes"] < 0,
                        0,
                        real_data_i_temp["Amplitudes"],
                    )
                    .detach()
                    .cpu()
                    .numpy()[0],
                },
                columns=["station", "azimuth", "takeoff", "polarity", "amplitudes"],
            )
            df_stat_obs = df_stat_obs[
                (df_stat_obs["polarity"] != 0)
                & (df_stat_obs["azimuth"] != 0)
                & (df_stat_obs["takeoff"] != 0)
            ]

            y_hat = SDR_pred[0].cpu()
            df_jacknife.loc[zero_staz] = {
                "strike": y_hat[0].item(),
                "dip": y_hat[1].item(),
                "rake": y_hat[2].item(),
                "kagan": KaganLoss(SDR_pred, y_hat_complete)["Kagan"].detach().cpu().item(),
            }

        az = real_data_i["az"].detach().cpu().numpy()[0]
        ih = real_data_i["ih"].detach().cpu().numpy()[0]
        pol = real_data_i["Polarities"].detach().cpu().numpy()[0]

        df_stat_obs = pd.DataFrame(
            data={
                "station": df_stations.staz.values,
                "azimuth": az,
                "takeoff": ih,
                "polarity": pol,
                "amplitudes": torch.where(
                    real_data_i["Amplitudes"] < 0,
                    0,
                    real_data_i["Amplitudes"],
                )
                .detach()
                .cpu()
                .numpy()[0],
            },
            columns=["station", "azimuth", "takeoff", "polarity", "amplitudes"],
        )
        df_stat_obs = df_stat_obs[
            (df_stat_obs["polarity"] != 0)
            & (df_stat_obs["azimuth"] != 0)
            & (df_stat_obs["takeoff"] != 0)
        ]

        rows, cols = 1, 2

        fig, axes = mplstereonet.subplots(
            projection="equal_area",
            figsize=[8 * cols, 10 * rows],
            nrows=rows,
            ncols=cols,
        )
        axes = np.atleast_1d(axes)

        fig, ax = plot_focal_mechanisms_pt(
            df_jacknife.loc[~df_jacknife.index.astype(str).str.contains("Complete")],
            df_stat_obs,
            alpha=0.21,
            line_style="gray",
            auxiliary_line_style="gray",
            title=f"Jackknife DeepFoc \n {name_event[idx_real_data]} \n \n",
            ax=axes.flatten()[0],
            use_color=True,
            jacknife=True,
        )

        fig, ax = plot_focal_mechanisms_pt(
            df_jacknife.loc[df_jacknife.index.astype(str).str.contains("Complete")],
            df_stat_obs,
            linewidth=3.0,
            alpha=1.0,
            line_style="k-",
            auxiliary_line_style="k-",
            title=f"Jackknife DeepFoc \n {name_event[idx_real_data]} \n \n",
            ax=axes.flatten()[0],
            use_color=True,
            jacknife=True,
        )

        fig, ax = plot_focal_mechanisms_pt(
            df_jacknife_fpfit[~df_jacknife_fpfit.index.str.contains("Complete")],
            df_stat_obs,
            alpha=0.21,
            line_style="gray",
            auxiliary_line_style="gray",
            title=f"Jackknife FPFIT \n {name_event[idx_real_data]} \n \n",
            ax=axes.flatten()[1],
            use_color=True,
            jacknife=False,
        )

        fig, ax = plot_focal_mechanisms_pt(
            df_jacknife_fpfit.loc[df_jacknife_fpfit.index.str.contains("Complete")],
            df_stat_obs,
            linewidth=3.0,
            alpha=1.0,
            line_style="k-",
            auxiliary_line_style="k-",
            title=f"Jackknife FPFIT \n {name_event[idx_real_data]} \n \n",
            ax=axes.flatten()[1],
            use_color=True,
            jacknife=True,
        )

        fig.savefig(
            save_noise_test / f"jacknife_{name_event[idx_real_data]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        img1_path = save_noise_test / f"jacknife_{name_event[idx_real_data]}.png"
        img2_path = (
            save_dir_cp_comparisons / f"real_model_fpfit_{name_event[idx_real_data]}.png"
        )

        img1 = mpimg.imread(img1_path)
        img2 = mpimg.imread(img2_path)

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[w1, w2])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        ax1.imshow(img1)
        ax1.axis("off")

        ax2.imshow(img2)
        ax2.axis("off")

        fig.tight_layout()
        fig.savefig(
            save_noise_test / f"combined_plot_{name_event[idx_real_data]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # # # # -----------------------------------------------------------------------------
    # SYNTHETIC DATA GENERATION AND ZERO FACTOR ANALYSIS
    #
    # This section generates a synthetic dataset using the FocalOnlineDataset class, which
    # produces a batch of synthetic focal mechanism data including spatial coordinates,
    # focal mechanism angles (strike, dip, rake), and auxiliary fields (azimuth,
    # takeoff angles, polarities). After generating the batch, we analyze the effect
    # of different "zero factors" applied during preprocessing on the model's predictions
    # and compare them with FPfit results.
    # The "zero factor" controls the level of zero noise applied during preprocessing.
    #
    # For each zero factor value, we:
    #   - Preprocess the synthetic batch.
    #   - Compute model predictions and calculate the Kagan loss (a metric for focal mechanism differences).
    #   - Run FPfit inversion on the preprocessed data.
    #   - Compare FPfit results with the model's predictions using histograms.
    #

    # -----------------------------------------------------------------------------
    # Create a synthetic dataset using FocalOnlineDataset.
    # -----------------------------------------------------------------------------

    histogram_plots = save_dir / "Plots" / "Histogram"
    histogram_plots.mkdir(exist_ok=True, parents=True)

    syntetich_generator = FocalOnlineDataset(
        xyz_boundary=args.xyz_boundary,
        sdr_boundary=args.sdr_boundary,
        xyz_steps=args.xyz_steps,
        sdr_steps=args.sdr_steps,
        exclude_strike=args.exclude_strike,
        exclude_dip=args.exclude_dip,
        exclude_rake=args.exclude_rake,
        velmod_file=args.velmod_file,
        stations_file=args.stations_file,
        input_dir=input_dir,
        batch_size=2048,
        generator=torch.Generator(device=device),
        device=device,
        xyz_translation_step_percs=None,
        float_type=args.float_type,
        polygon=polygon_zone,
    )

    # Set a fixed seed for reproducibility in synthetic data generation.
    syntetich_generator.set_seed(-2)
    # Generate a batch of synthetic data. The batch is a dictionary containing:
    #   - "XYZ": spatial coordinates (N x 3 tensor)
    #   - "SDR": focal mechanism angles (N x 3 tensor; strike, dip, rake)
    #   - "az": azimuth values for each station
    #   - "ih": takeoff angles for each station
    #   - "Polarities": polarity values (used in FPfit inversion and the model)
    #   - "Amplitudes": amplitude values (used only in the model)
    batch = syntetich_generator.gen_batch()

    # -----------------------------------------------------------------------------
    # Build a DataFrame from the generated batch.
    # -----------------------------------------------------------------------------
    # Extract spatial coordinates and focal mechanism angles from the batch.
    df_batch = pd.DataFrame(
        data={
            "x": batch["XYZ"][:, 0].detach().cpu().numpy(),  # X-coordinate values
            "y": batch["XYZ"][:, 1].detach().cpu().numpy(),  # Y-coordinate values
            "z": batch["XYZ"][:, 2].detach().cpu().numpy(),  # Z-coordinate values
            "strike": batch["SDR"][:, 0].detach().cpu().numpy(),  # Strike angle
            "dip": batch["SDR"][:, 1].detach().cpu().numpy(),  # Dip angle
            "rake": batch["SDR"][:, 2].detach().cpu().numpy(),  # Rake angle
        }
    )

    # Add additional columns for azimuth ('az_cols') and takeoff angles ('ih_cols').
    df_batch.loc[:, az_cols] = batch["az"].detach().cpu().numpy()
    df_batch.loc[:, ih_cols] = batch["ih"].detach().cpu().numpy()

    # -----------------------------------------------------------------------------
    # Initialize dictionaries and DataFrame for storing results.
    # -----------------------------------------------------------------------------
    # These will hold Kagan loss metrics for three methods:
    #   - "DeepFoc": Loss from the model's raw predictions.
    #   - "FPFIT": Loss computed from FPfit inversion results.
    #   - "DeepFoc_on_fpfit": Loss from model predictions only for events where FPfit found a solution.
    results_batch = {"FPFIT": {}, "DeepFoc": {}, "DeepFoc_on_fpfit": {}}

    # Define a set of zero factor values to test.
    zero_factors = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    multicols = pd.MultiIndex.from_product(
        [zero_factors, ["Means+std", "\u2264 20 Kagan"]], names=["Azzeramento", "Metric"]
    )
    df_results = pd.DataFrame(
        index=["FPFIT", "DeepFoc", "DeepFoc_on_fpfit"], columns=multicols
    )
    # Define a color palette for the histograms.
    color = sns.color_palette("tab10")
    # -----------------------------------------------------------------------------
    # Loop over different zero factors to assess performance.
    # -----------------------------------------------------------------------------
    for zero_factor in zero_factors:
        # Evaluate without tracking gradients.
        with torch.no_grad():
            # Create a deep copy of the original synthetic batch to avoid modifying it.
            temp_batch = deepcopy(batch)
            # Preprocess the batch with a given zero factor (affects data zero noise).
            temp_batch = model.preprocessing(temp_batch, zero_factor)
            # Append station positions to the batch by repeating x and y station coordinates.
            temp_batch["staz_pos"] = torch.stack(
                [
                    xstaz_tensor.repeat(len(temp_batch["Amplitudes"]), 1),
                    ystaz_tensor.repeat(len(temp_batch["Amplitudes"]), 1),
                ],
                dim=2,
            ).to(device)
            # Forward pass: compute the model outputs for the preprocessed batch.
            outputs = model(temp_batch)
            kagan_values = KaganLoss_nored(model.predict_angles(outputs), batch["SDR"])[
                "Kagan"
            ]
            # Store the loss values for this zero factor.
            results_batch["DeepFoc"][zero_factor] = kagan_values.cpu().numpy()
            # Compute the mean Kagan loss for the model predictions.
            kagan_loss = (
                np.mean(kagan_values.cpu().numpy()),
                np.std(kagan_values.cpu().numpy(), ddof=1),
                (torch.where(kagan_values <= 20, 1, 0).sum() / len(kagan_values)).item(),
            )

        # -----------------------------------------------------------------------------
        # FPfit Inversion on Preprocessed Data
        # -----------------------------------------------------------------------------
        # Create a copy of the DataFrame with synthetic data.
        df_temp = deepcopy(df_batch)
        # Update the polarity columns with values from the preprocessed batch.
        df_temp.loc[:, pol_cols] = temp_batch["Polarities"].detach().cpu().numpy()
        # Extract focal mechanism angles (strike, dip, rake) as a NumPy array.
        sdr_temp = df_temp[["strike", "dip", "rake"]].values
        # Run the FPfit inversion algorithm. This function returns a dictionary where each key corresponds
        # to an event and its value is a list of FPfit solutions (or None if no solution was found).
        result_batch = fpyfit(
            df_temp,
            df_stations,
            latref_lonref=[latref, lonref],
            file_name=f"{fpfit_folder}.prt",
            max_threads=64,
        )
        # Print diagnostic info: count events with multiple solutions and events with no solution.
        print(
            "Multiple solution:",
            np.sum(
                [1 if v is not None and len(v) > 1 else 0 for v in result_batch.values()]
            ),
        )
        print(
            "No solution:", np.sum([1 if v is None else 0 for v in result_batch.values()])
        )
        # Initialize a list to hold the average Kagan loss for each event as computed by FPfit.
        fpfit_result = []
        # Loop through each event's FPfit result.
        for i_event, v_rb in result_batch.items():
            # If no FPfit solution was found, store None.
            if v_rb is None:
                fpfit_result.append(None)
                continue
            # Extract the event index from the event's key (assumes key format includes the index as the last part).
            i_rb = int(i_event.split("_")[-1])
            # Compute the Kagan loss for each FPfit solution for this event by comparing it with the true SDR.
            kagan_angle_fpfit = [
                get_kagan_angle(v["strike"], v["dip"], v["rake"], *sdr_temp[i_rb])
                for v in v_rb
                if v is not None
            ]
            # Store the mean FPfit Kagan loss for the event.
            fpfit_result.append(np.mean(kagan_angle_fpfit))

        # Save FPfit results for the current zero factor.
        results_batch["FPFIT"][zero_factor] = fpfit_result
        # For events where FPfit provided a solution, store the corresponding model loss values.
        results_batch["DeepFoc_on_fpfit"][zero_factor] = results_batch["DeepFoc"][
            zero_factor
        ][[fr is not None for fr in fpfit_result]]
        # Filter out None values from FPfit results.
        fpfit_good_results = [fr for fr in fpfit_result if fr is not None]
        fpfit_good_results_less20 = np.sum(
            np.where(np.array(fpfit_good_results) <= 20, 1, 0)
        ) / len(fpfit_good_results)
        model_on_fpfit_less20 = np.sum(
            np.where(results_batch["DeepFoc_on_fpfit"][zero_factor] <= 20, 1, 0)
            / len(results_batch["DeepFoc_on_fpfit"][zero_factor])
        )
        print(f"Zero factor: {zero_factor}")
        print(
            f"Kagan Loss FPYFIT: {np.mean(fpfit_good_results)} \u00b1 {np.std(fpfit_good_results, ddof=1)}, total elements: {len(fpfit_good_results)}"
        )
        print(f"Kagan Loss model: {kagan_loss[0].item()} \u00b1 {kagan_loss[1].item()}")

        df_results.loc["FPFIT", (zero_factor, "Means+std")] = (
            f"{np.mean(fpfit_good_results):.3f} \u00b1 {np.std(fpfit_good_results, ddof=1):.3f}"
        )
        df_results.loc["DeepFoc", (zero_factor, "Means+std")] = (
            f"{kagan_loss[0].item():.3f} \u00b1 {kagan_loss[1].item():.3f}"
        )
        df_results.loc["DeepFoc_on_fpfit", (zero_factor, "Means+std")] = (
            f"{np.mean(results_batch['DeepFoc_on_fpfit'][zero_factor]):.3f} \u00b1 {np.std(results_batch['DeepFoc_on_fpfit'][zero_factor], ddof=1):.3f}"
        )
        df_results.loc["FPFIT", (zero_factor, "\u2264 20 Kagan")] = (
            f"{fpfit_good_results_less20*100:.3f} %"
        )
        df_results.loc["DeepFoc", (zero_factor, "\u2264 20 Kagan")] = (
            f"{kagan_loss[2]*100:.3f} %"
        )
        df_results.loc["DeepFoc_on_fpfit", (zero_factor, "\u2264 20 Kagan")] = (
            f"{model_on_fpfit_less20*100:.3f} %"
        )

    # Display and save the summary results of Kagan losses across all zero factors.
    display(df_results)
    df_results.to_csv(save_dir / "confronto_fpfit.csv")

    selected_zero_factors = [0.2, 0.5, 0.8]
    n_factors = len(selected_zero_factors)
    n_cols = 3
    n_rows = np.ceil(n_factors / n_cols).astype(int)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=False,
        sharey=True,
    )

    axes = axes.flatten()
    xticks = np.arange(0, 120, 20)
    for i, zero_factor in enumerate(selected_zero_factors):
        fpfit_data = [
            rb for rb in results_batch["FPFIT"][zero_factor] if rb is not None
        ]
        model_on_fpfit_data = results_batch["DeepFoc_on_fpfit"][zero_factor]

        sns.histplot(
            fpfit_data,
            kde=True,
            bins=60,
            color=color[2],
            edgecolor=color[2],
            label="FPFIT",
            ax=axes[i],
            stat="density",
            alpha=0.5,
        )
        sns.histplot(
            model_on_fpfit_data,
            kde=True,
            bins=60,
            color=color[4],
            edgecolor=color[4],
            label="DeepFoc",  # on fpfit",
            ax=axes[i],
            stat="density",
            alpha=0.5,
        )

        axes[i].set_title(f"Zero factor: {zero_factor}")
        axes[i].set_xticks(xticks)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=3.0)
    fig.savefig(
        histogram_plots
        / f"histogram_Mean_FPFIT_fpfit_vs_model_on_fpfit_overlay.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# %%
