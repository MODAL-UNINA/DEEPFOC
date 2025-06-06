#!/usr/bin/env python
# -*-coding:utf-8 -*-

# %%   IMPORTS

import os
import yaml
import shutil
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import trange
import matplotlib.pyplot as plt
from IPython.display import display
from fastcore.all import dict2obj, obj2dict, L
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as tmp

# Set the current working directory to the script's directory
os.chdir(Path(__file__).parent.resolve())  
import utils
import utils.model as md
from utils.loss import (
    KaganAngle,
    MomentMetrics,
    FocalMetrics,
    StereoDiscrepancyLoss,
)
from utils.online_dataset import FocalOnlineDataset
from utils.gen_utils import read_velmod_data, read_stations_data

from fpyfit.fpyfit import check_and_gclc

if __name__ == "__main__":
    args = dict2obj(
        {
            # Id of the GPUs to be used
            "id_device": "0", # e.g. "0,1" for 2 gpus 
            # Set the seed for reproducibility. Selecting the seed slows down the training of the model
            "seed": None,
            # Name of the model to be used
            "model_name": "AmplitudePolaritiesModel",
            # Directory to save the results
            "save_dir": "prova",
            # File with the stations
            "stations_file": "stations_campi_flegrei.dat",
            # File with the velocity model
            "velmod_file": "velmod_campi_flegrei.dat",
            # **Setting Dataset**
            # Limits in xyz coordinates of the area to be trained on (km)
            "xyz_boundary": [-4.5, 4.0, -4.0, 3.0, 2.0, 3.0],
            # Polygon within which to generate points and train the model. If None, points are generated based on xyz_boundary
            "polygon_zone": "Zone 1",
            # Maximum and minimum values in which to scale the coordinates (xmin, xmax, ymin, ymax, zmin, zmax)
            "scaled_parameters": [0, 1, 0, 1, 0, 1],
            # Boundary of the SDR (strike, dip, rake)
            "sdr_boundary": [0, 360, 0, 90, -90, 90],
            # Steps of the dataset
            "xyz_steps": [0.1, 0.1, 0.05],
            # Steps of the SDR dataset
            "sdr_steps": [1, 1, 1],
            # Excluded strikes
            "exclude_strike": [360],
            # Excluded dips
            "exclude_dip": [0, 90],
            # Excluded rakes
            "exclude_rake": [-180, 0, 180],
            # Float type
            "float_type": "float64",
            # **Setting Training**
            # Batch size
            "batch_size": 1024,
            # Number of epochs
            "epochs": 20000,
            # Port for communication between processes (default: 12355)
            "port": 12355,
            # Number of batches per epoch
            "batch_per_epoch": 100,
            # Print epoch interval for training
            "print_epoch": 25,
            # Early stopping patience
            "patience": 1000,
            # Learning rate
            "lr": 1e-4,
            # Weight decay for the optimizer (L2 regularization)
            "weight_decay": 1e-6,
            # **Setting Preprocessing**
            # Maximum percentage of zeroing for Polarities and Amplitudes
            "zeroing": [
                0.7,
                0.9,
            ],
            # Total concentration of beta distribution
            "beta_total_concentration": 4,
            # Switch sign for the Polarities
            "switch_sign": 0.0,
            # Normal noise for the Amplitudes
            "normal_noise": 0.25,
            # **Setting Loss and Metrics**
            # Metrics to be used as loss. To use more than one metric, separate the metrics with a +
            "loss": "Kagan+MSE",
            # Metrics with which to evaluate training at the end of the epoch
            "metrics": "Kagan+Discrepancy+Moment+FocalMetrics",
        }
    )

    # Checks if a seed has been specified via program arguments
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Convert L list to list
    for key, value in args.items():
        if isinstance(value, L):
            setattr(args, key, list(value))

# %%
if __name__ == "__main__":
    # Set the CUDA device to the specified GPU ID for computation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_device)

    # Get the absolute path of the current script directory
    scripts_dir = Path().resolve()
    root_dir = scripts_dir.parent  # Assuming the script is in a subdirectory

    # Define the directory to save output results, creating it if necessary
    save_dir = scripts_dir / "MODELS" / args.save_dir
    save_dir.mkdir(
        exist_ok=False, parents=True
    )  # Prevent overwriting if it already exists

    # Define dataset directory paths for input data
    data_dir = root_dir / "data"
    input_dir = data_dir / "INPUT"

    # Load station data from a CSV file and clean up missing values
    stations = pd.read_csv(input_dir / args.stations_file, sep=" ", header=None).dropna(
        axis=1
    )
    # Assign meaningful column names
    stations.columns = ["name", "lat", "lon", "elev"]
    # Set station names as the index for easier access
    stations.set_index("name", inplace=True)

    # Read velocity model data from a file
    latref, lonref, v, d = read_velmod_data(
        input_dir / args.velmod_file, np.dtype("float64")
    )

    # Read station coordinates and elevation data from another file
    staz, lats, lons, elev = read_stations_data(
        input_dir / args.stations_file, np.dtype("float64")
    )

    # Perform coordinate transformation and checks between reference and station locations
    ystaz, xstaz = check_and_gclc(latref, lonref, lats, lons)

    # Convert station coordinates to PyTorch tensors for further computation
    xstaz_tensor = torch.tensor(xstaz)
    ystaz_tensor = torch.tensor(ystaz)

    # Set the number of features (stations) in the arguments
    args.features_size = len(stations)

    # Checks if a specific polygon zone name has been provided in the program arguments
    if args.polygon_zone is not None:
        # Load the GeoJSON file containing definitions of various zones.
        zones_gpd = gpd.read_file(input_dir / "Campi_Flegrei_zones.geojson")
        # Set the 'Zone' column as the index for easy lookup of specific zones
        zones_gpd = zones_gpd.set_index("Zone")
        # Retrieve the geometry of the specified polygon zone from the loaded GeoJSON data
        polygon_zone = zones_gpd.loc[args.polygon_zone, "geometry"]
        # Extract the exterior coordinates of the polygon geometry and convert them to a NumPy array
        polygon = np.array(polygon_zone.exterior.coords.xy).T

        # The following checks adjust the 'xyz_boundary' parameters to match the bounding box of the selected polygon.
        # This ensures that even when using a polygon, the initial generation steps might be constrained by its minimal bounding box.
        # Check if the current x_min in args.xyz_boundary is less than the polygon's minimum x coordinate
        if args.xyz_boundary[0] < polygon[:, 0].min():
            # If it is, update args.xyz_boundary to the polygon's minimum x
            args.xyz_boundary[0] = float(polygon[:, 0].min())
            # Print a message indicating the adjustment was made based on the polygon
            print(
                f"xyz_boundary[0] (x_min) set to {polygon[:, 0].min()} because of polygon"
            )
        # Check and adjust the maximum y coordinate
        if args.xyz_boundary[1] > polygon[:, 0].max():
            args.xyz_boundary[1] = float(polygon[:, 0].max())
            print(
                f"xyz_boundary[1] (x_max) set to {polygon[:, 0].max()} because of polygon"
            )
        # Check and adjust the maximum x coordinate similarly
        if args.xyz_boundary[2] < polygon[:, 1].min():
            args.xyz_boundary[2] = float(polygon[:, 1].min())
            print(
                f"xyz_boundary[2] (y_min) set to {polygon[:, 1].min()} because of polygon"
            )
        # Check and adjust the minimum y coordinate
        if args.xyz_boundary[3] > polygon[:, 1].max():
            args.xyz_boundary[3] = float(polygon[:, 1].max())
            print(
                f"xyz_boundary[3] (y_max) set to {polygon[:, 1].max()} because of polygon"
            )
        
    # If no polygon zone was specified in the arguments (args.polygon_zone is None)
    else:
        # Set polygon_zone to None, indicating that no specific polygon is used to constrain point generation
        polygon_zone = None

    # Save the arguments to a YAML file for reproducibility
    with open(save_dir / "args.yml", "w") as f:
        yaml.safe_dump(obj2dict(args), f)

# %%


# Define the function to compute loss and metrics for a batch
def compute_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_dict: Dict[str, Callable],
    metrics_operator_dict: Dict[str, Callable],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the total loss and metrics for a single batch.

    This function takes the batch inputs, passes them through the model to get
    outputs, calculates the individual losses and metrics specified in the
    `loss_dict` and `metrics_operator_dict` configurations. It sums the
    individual losses to get the total loss used for backpropagation and collects
    the scalar values of individual losses and metrics into a results dictionary.

    Args:
        model: The PyTorch model to be evaluated, typically wrapped for distributed training.
        batch: A dictionary containing the input and target tensors for the current batch in 'XYZ' and 'SDR' keys.
        loss_dict: Dictionary with the loss functions to apply, mapped by name.
        metrics_operator_dict: Dictionary with the metric operators to calculate, mapped by name.

    Returns:
        A tuple containing:
        - loss (torch.Tensor): The scalar tensor representing the total loss for the batch.
        - results (dict): A dictionary containing the numerical values (float) of the
            individual losses and metrics calculated for the batch.
    """

    # Pass the input batch through the model to get raw outputs
    outputs = model(batch)
    # Convert the model's raw outputs into predicted SDR angles (Strike, Dip, Rake).
    # The `.module` is accessed because the model is wrapped in DistributedDataParallel.
    sdr_pred = model.module.predict_angles(outputs)
    # Calculate the target SDR representation (true angles) in the format used by the model for raw outputs.
    targets_trig = model.module.compute_trig_targets(batch["SDR"])

    # Initialize a dictionary to store the results of losses and metrics for this batch.
    results = {}
    # Calculate the total loss for backpropagation.
    # Initialize the total loss as a scalar tensor on zero, on the same device as the input data.
    loss = torch.tensor(0.0).to(batch["XYZ"].device)

    # The 'if' conditional blocks check which losses/metrics have been configured.
    # and calculate and add them accordingly.

    # Check if MSE loss is configured in loss_dict
    if "MSE" in loss_dict:
        # Calculate the MSE loss between the model outputs and the trig targets.
        mse_loss = loss_dict["MSE"](outputs, targets_trig)
        # Add the MSE loss to the total loss
        loss += mse_loss
        # Store the scalar (detached) value of the MSE loss in the results dictionary.
        results["MSE"] = mse_loss.detach().item()

    # Check if Kagan loss is configured in loss_dict
    if "Kagan" in loss_dict:
        # Calculate the Kagan loss using the predicted and true SDR angles.
        kagan_loss = loss_dict["Kagan"](sdr_pred, batch["SDR"])
        # Add the Kagan loss to the total loss
        loss += kagan_loss["Kagan"]
        # Store the scalar (detached) value of the Kagan loss in the results dictionary.
        results["Kagan"] = kagan_loss["Kagan"].detach().item()

    # Check if Discrepancy loss is configured in loss_dict
    if "Discrepancy" in loss_dict:
        # Calculate the Discrepancy loss using the predicted SDR angles and the batch data.
        discrepancy_loss = loss_dict["Discrepancy"](sdr_pred, batch)
        # Add the Discrepancy loss to the total loss
        loss += discrepancy_loss
        # Store the scalar (detached) value of the Discrepancy loss in the results dictionary.
        results["Discrepancy"] = discrepancy_loss.detach().item()

    # The following 'if' blocks handle the calculation of metrics that are not used directly for the total loss.

    # Check if "Moment" type metrics are configured and if the MomentMetrics operator has metrics to compute.
    if len(metrics_operator_dict["Moment"].metrics) > 0:
        # Calculate the "Moment" metrics using the predicted and true SDR angles.
        metrics = metrics_operator_dict["Moment"](sdr_pred, batch["SDR"])
        # Store each scalar (detached) value of the "Moment" metrics in the results dictionary.
        for metric, value_metric in metrics.items():
            results[metric] = value_metric.detach().item()
    # Check if "FocalMetrics" type metrics are configured and if the FocalMetrics operator has measures to compute.
    if len(metrics_operator_dict["FocalMetrics"].measure) > 0:
        # Calculate the "FocalMetrics" metrics using the predicted and true SDR angles.
        metrics = metrics_operator_dict["FocalMetrics"](sdr_pred, batch["SDR"])
        # Store each scalar (detached) value of the "FocalMetrics" metrics in the results dictionary.
        for metric, value_metric in metrics.items():
            results[metric] = value_metric.detach().item()

    # Return: Returns the total loss (scalar torch.Tensor) and a dictionary of results (dict with floats).
    return loss, results


def setup(rank: int, world_size: int, port: int = 12355, seed: int = None) -> None:
    """
    Initialize the distributed training environment.

    Args:
        rank (int): The rank of the current process in the distributed group.
        world_size (int): Total number of processes participating in distributed training.
        port (int, optional): Port for communication between processes. Defaults to 12355.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    This function sets up the distributed environment using NCCL as the backend,
    ensures all processes communicate properly, and initializes CUDA for the given rank.
    """

    torch.autograd.set_detect_anomaly(True)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # Set the master node address and communication port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Initialize the process group using NCCL (optimized for GPU communication)
    dist.init_process_group(
        backend="nccl",  # Backend for distributed GPU communication
        rank=rank,  # Rank of the current process
        world_size=world_size,  # Total number of processes
        timeout=datetime.timedelta(
            hours=3
        ),  # Set a timeout for process synchronization
    )

    # Set a fixed random seed for reproducibility across processes
    if seed is not None:
        print(f"SEED: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Assign the process to the corresponding GPU
    torch.cuda.set_device(rank)


def cleanup() -> None:
    """
    Clean up the distributed training environment.

    This function ensures all processes synchronize before destroying the process group.
    """
    dist.barrier()  # Ensure all processes reach this point before shutting down
    dist.destroy_process_group()  # Properly terminate the process group


def run_train(rank: int, world_size: int, run_train_dict: dict) -> None:
    """
    Distributed training function for a deep learning model using PyTorch.

    Args:
        rank (int): The rank of the current process in the distributed training setup.
        world_size (int): The total number of processes participating in training.
        run_train_dict (dict): Dictionary containing training-related arguments and data.

    This function:
    - Sets up the distributed environment.
    - Initializes data generators and model.
    - Performs distributed training with loss calculations.
    - Saves model checkpoints and logs training progress.
    - Implements early stopping for efficient training.
    """

    # Extract training arguments from dictionary
    args = run_train_dict["args"]
    stop_training = torch.tensor([0]).to(rank)  # Stop flag for early stopping
    save_dir = run_train_dict["save_dir"]
    input_dir = run_train_dict["input_dir"]
    xstaz_tensor, ystaz_tensor = run_train_dict["tensor_staz"]

    print(f"Process {rank} avviato su GPU {rank}")

    # Initialize distributed training environment
    setup(rank, world_size, port=args.port, seed=args.seed)

    # Initialize data generators for training and validation

    # Create the training data generator
    train_generator = FocalOnlineDataset(
        xyz_boundary=args.xyz_boundary,  # spatial limits [xmin, xmax, ymin, ymax, zmin, zmax] (km) for sampling event locations
        sdr_boundary=args.sdr_boundary,  # orientation limits [strike_min, strike_max, dip_min, dip_max, rake_min, rake_max] (degrees)
        xyz_steps=args.xyz_steps,  # grid resolution in x, y, z for initial sampling (km)
        sdr_steps=args.sdr_steps,  # resolution for strike, dip, rake sampling (degrees)
        exclude_strike=args.exclude_strike,  # specific strike angles to skip (e.g., 360°)
        exclude_dip=args.exclude_dip,  # specific dip angles to skip (e.g., 0°, 90°)
        exclude_rake=args.exclude_rake,  # specific rake angles to skip (e.g., -180°, 0°, 180°)
        velmod_file=args.velmod_file,  # path to velocity model file used for travel-time calculations
        stations_file=args.stations_file,  # path to station metadata (names, lat/lon, elevation)
        input_dir=input_dir,  # base directory containing input files
        batch_size=args.batch_size,  # number of samples per batch
        generator=torch.Generator(
            device=rank
        ),  # PyTorch RNG tied to this GPU for reproducible shuffling
        device=rank,  # GPU device index for on-the-fly data generation
        xyz_translation_step_percs=None,  # no spatial jitter applied for training samples
        float_type=args.float_type,  # precision of generated tensors (e.g., float32, float64)
        polygon=run_train_dict[
            "polygon"
        ],  # optional geographic polygon to constrain sampling region
    )

    # Create the validation data generator
    val_generator = FocalOnlineDataset(
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
        batch_size=args.batch_size,
        generator=torch.Generator(device=rank),
        device=rank,
        xyz_translation_step_percs=[  # apply half-step spatial jitter for validation augmentation
            f / 2 for f in args.xyz_steps
        ],
        float_type=args.float_type,
        polygon=run_train_dict["polygon"],
    )

    # Seed the generators to ensure reproducible but distinct random streams across processes
    train_generator.set_seed(rank + 1)  # seed training generator with rank offset
    val_generator.set_seed(
        world_size + (rank + 1)
    )  # seed validation generator to avoid overlap with training streams

    # Load and initialize the model for training
    CustomModel = getattr(md, args.model_name)
    model = (
        CustomModel(
            n_stations=args.features_size,
            xyz_boundary=args.xyz_boundary,
            scaling_range=args.scaled_parameters,
        )
        .to(rank)
        .to(getattr(torch, args.float_type))
    )
    # Copy model architecture file for reference
    shutil.copy(utils.model.__file__, save_dir / "model.py")

    # Build the model by fetching one training batch and running it through the network

    # Generate a new batch of synthetic training data
    batch = train_generator.gen_batch()  

    # Assemble station coordinates tensor and attach to batch
    batch["staz_pos"] = (
        torch.stack(
            [
                # repeat station X coordinates for each sample in batch
                xstaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
                # repeat station Y coordinates for each sample in batch
                ystaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
            ],
            dim=2,             # stack along a new last dimension to get shape [batch, stations, 2]
        )
        .detach()             # detach from computation graph (no gradients needed)
        .to(rank)             # move tensor to the current GPU device
    )

    # Forward-pass the batch through the model
    model(batch)

    # If a geographic polygon is defined and this is the primary process, plot sample locations vs. polygon
    if run_train_dict["polygon"] is not None and rank == 0:
        # Extract X, Y coordinates of polygon exterior
        x, y = np.array(run_train_dict["polygon"].exterior.coords.xy)

        # Create a scatter plot of event hypocenters (red dots) and overlay polygon boundary
        fig, ax = plt.subplots()
        ax.scatter(batch["XYZ"][:, 0].cpu(), batch["XYZ"][:, 1].cpu(), s=1, c="red")  # event X/Y points
        ax.plot(x, y)                               # polygon outline
        ax.set_title("Polygon")                     # plot title
        ax.set_xlabel("X")                          # X-axis label (km)
        ax.set_ylabel("Y")                          # Y-axis label (km)
        plt.axis("equal")                           # ensure equal aspect ratio

        # Save figure to disk and close to free memory
        fig.savefig(save_dir / "polygon.png")
        plt.close()

    # Synchronize all processes before proceeding (ensures everyone waits here)
    dist.barrier(device_ids=[rank])

    # Wrap model for distributed training
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
    )

    # Define optimizer and loss functions for training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Define loss functions and evaluation metrics
    losses_list = args.loss.split("+")
    loss_dict = {}
    if "Kagan" in losses_list:
        loss_dict["Kagan"] = KaganAngle(reduction="mean", normalize=True).to(rank)
    if "MSE" in losses_list:
        loss_dict["MSE"] = torch.nn.MSELoss(reduction="mean").to(rank)
    if "Discrepancy" in losses_list:
        loss_dict["Discrepancy"] = StereoDiscrepancyLoss(
            reduction="mean", normalize=True
        ).to(rank)

    metrics_operator_dict = {}
    metrics_list = args.metrics.split("+")
    if "Kagan" in metrics_list and not "Kagan" in loss_dict:
        metrics_operator_dict["Kagan"] = KaganAngle(
            reduction="mean", normalize=True
        ).to(rank)
    if "Discrepancy" in metrics_list and not "Dipcrepancy" in loss_dict:
        metrics_operator_dict["Discrepancy"] = StereoDiscrepancyLoss(
            reduction="mean", normalize=True
        ).to(rank)
    if "Moment" in metrics_list and not "Moment" in loss_dict:
        metrics_operator_dict["Moment"] = MomentMetrics(
            metrics=[
                "relative_frobenius",
                "angular_distance",
                "cosine_similarity",
            ],
            reduction="mean",
        ).to(rank)
    if "FocalMetrics" in metrics_list and not "FocalMetrics" in loss_dict:
        metrics_operator_dict["FocalMetrics"] = FocalMetrics(measure=["mse", "mae"]).to(
            rank
        )

    # Initialize variables for training progress and early stopping
    best_val_loss = torch.inf
    wait = 0  # Counter for early stopping

    # Initialize progress bar for epochs
    if rank == 0:
        tqdm_range = trange(1, args.epochs + 1, desc="Epochs")
    else:
        tqdm_range = range(1, args.epochs + 1)

    # Define number of batches for validation
    batch_per_epoch_val = args.batch_per_epoch // 4 + 1

    # Initialize a dictionary to store metric time series
    metrics_dict = {"Loss": []}  # always track training loss

    # Add entries for each loss component returned in loss_dict
    for key in loss_dict:
        metrics_dict[key] = []

    # If the Moment metrics operator has defined metrics, add them
    if len(metrics_operator_dict["Moment"].metrics) > 0:
        for metric in metrics_operator_dict["Moment"].metrics:
            metrics_dict[metric] = []

    # If the FocalMetrics operator has defined measures, add them
    if len(metrics_operator_dict["FocalMetrics"].measure) > 0:
        for metric in metrics_operator_dict["FocalMetrics"].measure:
            metrics_dict[metric] = []

    # Create separate containers for training and validation metrics (deep copied to keep them independent)
    metrics_dict = {
        "Train": deepcopy(metrics_dict),
        "Validation": deepcopy(metrics_dict),
    }

    # Initialize running sums for each training metric to compute epoch-level averages
    running_epoch = {k: 0.0 for k in metrics_dict["Train"].keys()}  

    if rank != 0:
        del metrics_dict

    for epoch in tqdm_range:

        model.train()

        running_loss = deepcopy(running_epoch)
        running_val_loss = deepcopy(running_epoch)

        for i in range(args.batch_per_epoch):
            # Generate a batch of training data
            batch = train_generator.gen_batch()

            # Preprocess batch data for training
            batch = model.module.preprocessing(
                batch,
                args.zeroing,
                switch_sign=torch.tensor(args.switch_sign).to(rank),
                normal_noise=args.normal_noise,
                beta_total_concentration=args.beta_total_concentration,
            )
            # Set station coordinates for the batch
            batch["staz_pos"] = (
                torch.stack(
                    [
                        xstaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
                        ystaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
                    ],
                    dim=2,
                )
                .detach()
                .to(rank)
            )

            # Forward pass and loss computation
            loss, loss_results = compute_loss(
                model,
                batch,
                loss_dict,
                metrics_operator_dict,
            )

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss and metrics
            running_loss["Loss"] += loss.detach().item()
            for key_lr, value_lr in loss_results.items():
                if key_lr == "Kagan":
                    running_loss[key_lr] += value_lr * 120
                else:
                    running_loss[key_lr] += value_lr

        # Compute average loss for the epoch
        for key_rn in running_loss.keys():
            running_loss[key_rn] /= args.batch_per_epoch

        epoch_loss = running_loss["Loss"]

        # Synchronize and reduce losses across all ranks
        epoch_train_measures = torch.tensor(list(running_loss.values())).to(rank)
        dist.barrier()
        dist.reduce(epoch_train_measures, dst=0, op=dist.ReduceOp.AVG)

        # Update metrics dictionary for training data
        if rank == 0:
            for i_key_rn, key_rn in enumerate(running_loss.keys()):
                metrics_dict["Train"][key_rn].append(
                    epoch_train_measures[i_key_rn].item()
                )
                running_loss[key_rn] = epoch_train_measures[i_key_rn].item()
            # tqdm_range.set_postfix(running_loss)

        del epoch_train_measures

        # Print training progress
        if epoch % args.print_epoch == 0 and rank == 0:
            print("┌" + "-" * 50 + "┐")
            print(f"  Epoch [{epoch}/{args.epochs}], Average Loss: {epoch_loss:.5f}")
            for key_rn, value_rn in running_loss.items():
                if key_rn == "Loss":
                    continue
                print(f"\t {key_rn}: {value_rn:.5f}")

        # Validate the model on the validation dataset
        model.eval()

        # Disable gradient computation for validation
        with torch.no_grad():
            for _ in range(batch_per_epoch_val):
                # Generate a batch of validation data
                batch = val_generator.gen_batch()
                # Preprocess batch data for validation
                batch = model.module.preprocessing(
                    batch,
                    args.zeroing,
                    switch_sign=torch.tensor(args.switch_sign).to(rank),
                    normal_noise=args.normal_noise,
                    beta_total_concentration=args.beta_total_concentration,
                )
                # Set station coordinates for the batch
                batch["staz_pos"] = torch.stack(
                    [
                        xstaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
                        ystaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
                    ],
                    dim=2,
                )

                # Forward pass and loss computation for validation
                loss, loss_results = compute_loss(
                    model,
                    batch,
                    loss_dict,
                    metrics_operator_dict,
                )

                running_val_loss["Loss"] += loss.item()
                for key_lr, value_lr in loss_results.items():
                    if key_lr == "Kagan":
                        running_val_loss[key_lr] += value_lr * 120
                    else:
                        running_val_loss[key_lr] += value_lr

        for key_rvn in running_val_loss.keys():
            running_val_loss[key_rvn] /= batch_per_epoch_val

        avg_val_loss = running_val_loss["Loss"]

        # Synchronize and reduce validation losses across all ranks
        epoch_val_measures = torch.tensor(list(running_val_loss.values())).to(rank)
        dist.barrier()
        dist.reduce(epoch_val_measures, dst=0, op=dist.ReduceOp.AVG)

        # Update metrics dictionary for validation data
        if rank == 0:
            for i_key_rn, key_rn in enumerate(running_val_loss.keys()):
                metrics_dict["Validation"][key_rn].append(
                    epoch_val_measures[i_key_rn].item()
                )
                running_val_loss[key_rn] = epoch_val_measures[i_key_rn].item()

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                # Save the model with the best validation loss
                print(f"  Validation Loss improved: {avg_val_loss:.5f}")
                best_val_loss = avg_val_loss
                model.module.save_parameters_correctly(save_dir / "best_model.pth")
                wait = 0
            else:
                # Increment the early stopping counter
                wait += 1
                # Check if the patience limit has been reached
                if wait >= args.patience or epoch == args.epochs:
                    print(f"End of training at epoch {epoch}")
                    stop_training[0] = 1.0

            # Print validation progress
            if wait == 0 or epoch % args.print_epoch == 0:
                for key_rn, value_rn in running_val_loss.items():
                    if key_rn == "Loss":
                        continue
                    print(f"\t {key_rn}: {value_rn:.5f}")

        # Plot and save metrics at regular intervals
        if epoch % args.print_epoch == 0 and rank == 0:
            with open(save_dir / "metrics.pkl", "wb") as f:
                pickle.dump(metrics_dict, f)

            for key_md_plot in metrics_dict["Train"].keys():
                plt.figure(figsize=(10, 5))
                plt.title(f"{key_md_plot}")
                plt.plot(metrics_dict["Train"][key_md_plot], label="Train")
                plt.plot(metrics_dict["Validation"][key_md_plot], label="Validation")
                plt.legend()
                plt.savefig(save_dir / f"{key_md_plot}.png")
                plt.close()

                # Plot last 100 epochs
                plt.figure(figsize=(10, 5))
                plt.title(f"{key_md_plot} Last 100 epochs")
                plt.plot(metrics_dict["Train"][key_md_plot][-100:], label="Train")
                plt.plot(
                    metrics_dict["Validation"][key_md_plot][-100:], label="Validation"
                )
                plt.legend()
                plt.savefig(save_dir / f"{key_md_plot}_last100.png")
                plt.close()

            # Plotting metrics
            for key_md_plot in metrics_dict["Train"].keys():
                plt.figure(figsize=(10, 5))
                plt.title(f"{key_md_plot}")
                plt.legend()
                plt.savefig(save_dir / f"test_{key_md_plot}.png")
                plt.close()

                # Plotting moving average metrics
                plt.figure(figsize=(10, 5))
                plt.title(f"{key_md_plot}")
                plt.legend()
                plt.savefig(save_dir / f"test_{key_md_plot}_ma.png")
                plt.close()

        dist.barrier()
        dist.broadcast(stop_training, src=0)
        # Stop training if early stopping condition is met
        if stop_training.item() == 1:
            model.module.load_parameters_correctly(save_dir / "best_model.pth")
            break

    # Save the final model and metrics
    if rank == 0:
        plt.figure(figsize=(10, 5))
        plt.title("Final Loss")
        plt.plot(metrics_dict["Train"]["Loss"], label="Train Loss")
        plt.plot(metrics_dict["Validation"]["Loss"], label="Validation Loss")
        plt.legend()
        plt.savefig(save_dir / "Loss.png")
        plt.close()

        with open(save_dir / "metrics.pkl", "wb") as f:
            pickle.dump(metrics_dict, f)

    # Initialize the KaganAngle loss function for evaluating angular differences
    kaganangle = KaganAngle()

    # Set a fixed seed never used for validation to ensure reproducibility
    val_generator.set_seed(-1)

    # Generate a validation batch for testing the model
    batch = val_generator.gen_batch()

    # Preprocess the batch data for testing
    batch = model.module.preprocessing(
        batch,
        args.zeroing,
        switch_sign=torch.tensor(args.switch_sign).to(rank),
        normal_noise=args.normal_noise,
        beta_total_concentration=args.beta_total_concentration,
    )
    # Set station coordinates for the batch
    batch["staz_pos"] = torch.stack(
        [
            xstaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
            ystaz_tensor.unsqueeze(0).repeat(batch["XYZ"].size(0), 1),
        ],
        dim=2,
    )

    # Test the model on the validation dataset
    with torch.no_grad():
        outputs = model.module.predict_angles(model(batch))
        targets = batch["SDR"]

        kagan_loss = kaganangle(outputs, targets)
        df_results = pd.DataFrame(
            index=["Mean", "Std", "Min", "Max"], columns=kagan_loss.keys()
        )
        for key, value in kagan_loss.items():
            df_results.loc["Mean", key] = value.mean().item()
            df_results.loc["Std", key] = value.std().item()
            df_results.loc["Min", key] = value.min().item()
            df_results.loc["Max", key] = value.max().item()
        display(df_results)

    # Save the test results to a CSV file
    if rank == 0:
        df_results.to_csv(save_dir / "test_results.csv")

    cleanup()


# %%

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    n_gpus = torch.cuda.device_count()
    print(f"Avviando il training su {n_gpus} GPU...")

    run_train_dict = dict(
        args=args,
        save_dir=save_dir,
        input_dir=input_dir,
        tensor_staz=(xstaz_tensor, ystaz_tensor),
        polygon=polygon_zone,
    )
    tmp.spawn(run_train, args=(n_gpus, run_train_dict), nprocs=n_gpus, join=True)


# %%
