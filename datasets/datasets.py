import os

import numpy as np
import pandas as pd
from sklearn.datasets import (fetch_openml)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler)
from distortion_cues.config import GLOBAL_SEED


# Get the current working directory (main project folder)
project_dir = os.getcwd()


# #####__________ 3D Cubic_______________________________________________________________
def generate_cube_points(center, side_length, n_points, random_state=GLOBAL_SEED):
    """
    Generate uniformly distributed 3D points inside a cube.

    Parameters
    ----------
    center : tuple(float, float, float)
        The (x, y, z) coordinates of the cube center.
    side_length : float
        The length of the cube's side.
    n_points : int
        Number of points to sample uniformly within the cube.
    random_state : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing 3D coordinates.
    """
    rng = np.random.default_rng(random_state)
    half_side = side_length / 2.0

    points = rng.uniform(-half_side, half_side, size=(n_points, 3))
    points += np.array(center)


    return points


def generate_multi_cube_dataset(
    n_points_per_cube=500,
    cube_sizes=(5.0, 3.0, 1.5),
    cube_centers=((0, 0, 0), (10, 0, 0), (20, 10, 5)),
    outlier_cube_index=None,
    n_outliers=0,
    outlier_distance=5.0,
    normalize=True,
    random_state=GLOBAL_SEED
):
    """
    Generate multiple cubic 3D clusters, optionally injecting outliers.

    Parameters
    ----------
    n_points_per_cube : int
        Number of points per cube.
    cube_sizes : tuple of floats
        Side lengths of cubes.
    cube_centers : tuple of tuples
        Centers of cubes.
    outlier_cube_index : int | None
        Index of cube (0-based) where outliers are added. None = no outliers.
    n_outliers : int
        Number of outliers to generate.
    outlier_distance : float
        Distance multiplier for how far outliers are placed from the cube center.
    normalize : bool
        Whether to normalize the full dataset to [0, 1].
    random_state : int | None
        Random seed.

    Returns
    -------
    data : np.ndarray
        (N_total_points, 3) array (including outliers if any).
    labels : np.ndarray
        Integer class labels (0, 1, 2, ..., and -1 for outliers).
    outlier_mask : np.ndarray
        Boolean mask where True = outlier point.
    """
    if len(cube_sizes) != len(cube_centers):
        raise ValueError("cube_sizes and cube_centers must have the same length.")

    rng = np.random.default_rng(random_state)
    all_points, all_labels = [], []

    # --- Generate core cubes ---
    for idx, (size, center) in enumerate(zip(cube_sizes, cube_centers)):
        pts = generate_cube_points(center, size, n_points_per_cube, random_state=rng.integers(1e9))
        all_points.append(pts)
        all_labels.append(np.full(n_points_per_cube, idx))

    outlier_mask = np.zeros(sum(len(p) for p in all_points), dtype=bool)

    # --- Inject outliers (distance-controlled generation) ---
    if outlier_cube_index is not None and n_outliers > 0:

        target_cluster_points = all_points[outlier_cube_index]

        # Distance thresholds
        min_dist_from_target = 3   # must be at least this far from target cluster
        min_dist_from_others = 2   # must NOT get closer than this to other clusters

        generated = []
        attempts = 0
        max_attempts = 50000

        while len(generated) < n_outliers and attempts < max_attempts:
            attempts += 1

            # Generate a candidate point near the target center
            base_center = np.array(cube_centers[outlier_cube_index])
            base_size = cube_sizes[outlier_cube_index]

            candidate = base_center + rng.uniform(
                low=-base_size * outlier_distance,
                high=base_size * outlier_distance,
                size=3
            )

            # ---- Check distance to target cluster ----
            d_target = np.min(np.linalg.norm(target_cluster_points - candidate, axis=1))

            if d_target < min_dist_from_target:
                continue  # too close → not an outlier

            # ---- Check against other clusters (avoid landing inside) ----
            too_close_other = False
            for j, cluster_pts in enumerate(all_points):
                if j == outlier_cube_index:
                    continue
                d_other = np.min(np.linalg.norm(cluster_pts - candidate, axis=1))
                if d_other < min_dist_from_others:
                    too_close_other = True
                    break

            if too_close_other:
                continue  # too close to another cluster → reject

            # Passed all checks → keep this outlier
            generated.append(candidate)

        if len(generated) < n_outliers:
            print(f"Warning: only generated {len(generated)} outliers after {attempts} attempts.")

        generated = np.array(generated)

        # Store in structure
        all_points.append(generated)
        all_labels.append(np.full(len(generated), 999))  # outlier label

        outlier_mask = np.concatenate([
            np.zeros(sum(len(p) for p in all_points[:-1]), dtype=bool),
            np.ones(len(generated), dtype=bool)
        ])


    # --- Combine ---
    data = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    # --- Optional normalization ---
    if normalize:
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    return data, labels, outlier_mask

##___________end 3D Cubic dataset___________________________________________


def selected_dataset_dt(
    dataset, num_dim = 3, n_pts_per_gauss = 200, cluster_spacing=1.0, spread_factor=0.01, cube_sizes = None, cube_centers = None,
    outlier_cube_index = None, n_outliers = 0, outlier_distance=5.0
):

    """
    Load or generate a dataset for analysis.

    This function provides a unified interface for both synthetic and real-world datasets. 
    Synthetic datasets include Gaussian mixtures and tetrahedral cluster structures, while 
    real-world datasets include standard benchmarks (e.g., Iris, Digits, MNIST, Wine) and 
    several domain-specific datasets loaded from CSV files.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load. Options include:
        - Synthetic: "gaussian", "tetrahedron_eq", "tetrahedron_eq_1_far",
        "tetrahedron_eq_1_close", "tetrahedron_eq_2_close", "high_dim".
        - Benchmark: "iris", "digits", "wine", "breast", "har", "cifar", "mnist", "linnerud".
        - Domain-specific: "bank_superv", "hcv_superv", "adessex", "anurancalls", 
        "arabicdigit", "bng", "magic", "mozilla".
    num_dim : int, default=3
        Dimensionality of generated synthetic datasets (used for "high_dim").
    n_pts_per_gauss : int, default=200
        Number of samples per Gaussian cluster for synthetic datasets.
    cluster_spacing : float, default=1.0
        Controls spacing between cluster centers in Gaussian-based synthetic datasets.
    spread_factor : float, default=0.01
        Variance control for Gaussian clusters in synthetic datasets.

    Returns
    -------
    D : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    c : numpy.ndarray
        Labels (cluster IDs or class labels).
    dim : int
        Dimensionality of the dataset.
    output_size : int
        Same as `dim`, included for downstream compatibility.
    n_gauss : int
        Number of clusters (synthetic) or unique classes (real-world).

    Notes
    -----
    All datasets are returned in a consistent format, making the function 
    suitable for supervised, unsupervised, and visualization tasks.
    """


    if dataset in ['cubic','cubic_2_clusters' ,'cube_diff_size_diff_dist', 'cube_same_size_same_dist', 'cube_diff_size_same_dist', 'cube_same_size_diff_dist']:

        D, c,_ = generate_multi_cube_dataset(
        n_points_per_cube=n_pts_per_gauss,
        cube_sizes=cube_sizes,
        cube_centers=cube_centers,
        random_state=GLOBAL_SEED)

        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))

    
    
    elif dataset == "har":

        D, c = har_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))

    elif dataset == "mnist":
        D, c = MNIST()
        dim = D.shape[1]
        output_size = dim
        # breakpoint()
        n_gauss = len(np.unique(c))
    
    elif dataset in ["aedessex"]:

        data_path = os.path.join("datasets", "dataset_new_setup", "AedesSex.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)
        D, label = aedessex(dt)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c = label

    else:
        raise ValueError("Invalid dataset name.")

    return D, c, dim, output_size, n_gauss


##################################################################################################################################



def har_dt(sample_size=None, random_state=GLOBAL_SEED):
    """
    Load and preprocess the HAR dataset, retaining only specific activities.

    Parameters:
        sample_size (int): Number of samples to retain (optional).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Normalized features and filtered labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(
        project_dir, "datasets", "UCI_ HAR_ Dataset", "train"
    )

    # File paths
    x_train_path = os.path.join(datasets_folder, "X_train.txt")
    y_train_path = os.path.join(datasets_folder, "y_train.txt")

    # Load the data as NumPy arrays
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path)

    # Filter for specific labels (WALKING: 1, SITTING: 4, STANDING: 5,  Laying: 6)
    desired_labels = [1, 2, 3, 4, 5, 6]
    # desired_labels = [1, 3]# 3, 4, 5, 6]   # C_0 = 1226, C_1= 1073,  C_2= 986
    mask = np.isin(y_train, desired_labels)
    X_filtered = X_train[mask]
    y_filtered = y_train[mask]

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_filtered)

    # Remap the labels to consecutive integers
    label_mapping = {
        original: new for new, original in enumerate(desired_labels, start=0)
    }
    y_filtered = np.array([label_mapping[label] for label in y_filtered])
    # If sample_size is specified, perform stratified sampling
    if sample_size is not None and sample_size < len(y_filtered):
        X_normalized, _, y_filtered, _ = train_test_split(
            X_normalized,
            y_filtered,
            train_size=sample_size,
            stratify=y_filtered,
            random_state=random_state,
        )
    # breakpoint()
    # Count samples for each class
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")

        # Sort indices based on class labels
    sorted_indices = np.argsort(y_filtered)

    # Reorder dataset and labels
    X_sorted = X_normalized[sorted_indices]
    y_sorted = y_filtered[sorted_indices]
    # return X_normalized, y_filtered
    return X_sorted, y_sorted


def MNIST():
    mnist = fetch_openml("mnist_784", version=1, cache=True)
    X, y = (
        mnist.data.to_numpy()[::7],
        mnist.target.to_numpy()[::7],
    )  # X: images, y: labels
    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    return X_normalized, y

def aedessex(dt):
    X = dt.drop(columns=["class"])
    y = dt["class"]
    breakpoint()
    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return np.array(X_norm), np.array(y)
