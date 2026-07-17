from collections import namedtuple
import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import Normalize
from sklearn.metrics import pairwise_distances

from distortion_cues import config as cfg
from shapely.geometry import Polygon
import umap
from sklearn import manifold
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.colors import Normalize
from skimage.draw import line as rasterize_line
import time
from matplotlib.colors import ListedColormap, to_rgb
import colorcet as cc


def x_y_range(emb, margin=0.05):
    """
    Calculates the range of x and y values from the embedding with optional margin.

    Parameters:
        embedding (np.ndarray): 2D projection coordinates.
        margin (float): Extra margin to add to the min and max range (default is 5%).

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]: (x_range, y_range).
    """

    XYRange = namedtuple("XYRange", ["x_min", "x_max", "y_min", "y_max"])
    x_min, x_max = np.min(emb[:, 0]), np.max(emb[:, 0])
    y_min, y_max = np.min(emb[:, 1]), np.max(emb[:, 1])

    ###_____Margin_______________________________________
    margin = margin  # 5% margin (adjustable)
    x_min, x_max = np.min(emb[:, 0]), np.max(emb[:, 0])
    y_min, y_max = np.min(emb[:, 1]), np.max(emb[:, 1])

    # Expand the range by a percentage of the range
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    return XYRange(x_min, x_max, y_min, y_max)


####_____________________Delanay______________________________________________


def calculate_delanay_edge_length(hd_data, ld_data, tri_nodes):
    """
    Calculates the edge lengths of the Delaunay triangles in both high-dimensional and low-dimensional spaces.

    Parameters:
        hd_data (np.ndarray): High-dimensional data.
        ld_data (np.ndarray): 2D embedding (Projection) of the HD data.
        tri_nodes (np.ndarray): Indices of triangle vertices from Delaunay triangulation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Edge lengths in HD and LD for each triangle.
    """

    all_edges_length_ld = []
    all_edges_length_hd = []

    for simplex in tri_nodes:
        # Get the coordinates of the three triangle points
        pt_v1_LD, pt_v2_LD, pt_v3_LD = ld_data[simplex]  # Shape (3, 2)
        pt_v1_HD, pt_v2_HD, pt_v3_HD = hd_data[simplex]  # Shape (3, feature_dim)

        # Compute Euclidean distances between the three edges in Low Dimension
        edge_lengths_LD = [
            np.linalg.norm(pt_v1_LD - pt_v2_LD),
            np.linalg.norm(pt_v2_LD - pt_v3_LD),
            np.linalg.norm(pt_v3_LD - pt_v1_LD),
        ]

        # Compute Euclidean distances between the three edges in High Dimension
        edge_lengths_HD = [
            np.linalg.norm(pt_v1_HD - pt_v2_HD),
            np.linalg.norm(pt_v2_HD - pt_v3_HD),
            np.linalg.norm(pt_v3_HD - pt_v1_HD),
        ]

        all_edges_length_ld.append(edge_lengths_LD)
        all_edges_length_hd.append(edge_lengths_HD)

    return np.array(all_edges_length_hd), np.array(all_edges_length_ld)



##_________________ Interpolation_______________________________________


def create_matrix_determinant(V1, V2, P):
    """
    Constructs a 3x3 matrix using the given two vertices and a point.

    Parameters:
    - V1: Tuple (x1, y1) representing the first vertex.
    - V2: Tuple (x2, y2) representing the second vertex.
    - P: Tuple (xp, yp) representing the point.

    Returns:
    - A: 3x3 NumPy matrix with the specified elements.
    """
    A = np.array(
        [
            [V1[0], V2[0], P[0]],  # First row (x-coordinates)
            [V1[1], V2[1], P[1]],  # Second row (y-coordinates)
            [1, 1, 1],  # Third row (all ones)
        ]
    )

    det_A = np.linalg.det(A) / 2
    return A, det_A


def barycentric_interpolation(
    num_grid_points_inter,
    embedding,
    tri_delaunay,
    feature_values,
    x_y_range,
    bVertex_based=False,
    blog=False,
    bclamping=False,
):
    """
    Perform barycentric interpolation over a 2D domain using Delaunay triangulation.

    This function interpolates feature values defined either on triangle vertices or
    per-triangle (depending on `bVertex_based`) onto a regular grid of points. 
    The interpolation is done using barycentric coordinates with optional 
    logarithmic scaling and clamping.

    Parameters
    ----------
    num_grid_points_inter : int
        Number of grid points per axis used for interpolation (resolution of grid).
    
    embedding : ndarray
        Projection of high dimensional data.
    
    tri_delaunay : scipy.spatial.Delaunay
        Delaunay triangulation object containing simplices and vertices.
    
    feature_values : ndarray
        Values to interpolate.  
        - If `bVertex_based=True`, should be an array with one value per vertex.  
        - If `bVertex_based=False`, should be an array with shape (n_triangles, 3), 
          containing per-triangle feature values.
    
    x_y_range : object
        An object with attributes `x_min`, `x_max`, `y_min`, `y_max`, 
        defining the grid domain for interpolation.
    
    bVertex_based : bool, optional (default=False)
        If True, interpolate using feature values defined at vertices.  
        If False, interpolate using per-triangle feature values.
    
    blog : bool, optional (default=False)
        If True, apply logarithmic scaling to interpolated values before normalization.
    
    bclamping : bool, optional (default=False)
        If True, clamp interpolated values to the 1st–99th percentile 
        range before normalization.

    Returns
    -------
    intensity_interp_reshape : ndarray of shape (num_grid_points_inter, num_grid_points_inter)
        2D grid of interpolated intensity values. Values outside the convex hull 
        of the triangulation are set to NaN.

    Notes
    -----
    - Normalization rescales valid interpolated values into the [0, 1] range.  
    - If `blog=True`, logarithmic scaling is applied before normalization.  
    - If `bclamping=True`, extreme values are clipped to reduce the influence 
      of outliers.
    """

    x_min, x_max = x_y_range.x_min, x_y_range.x_max
    y_min, y_max = x_y_range.y_min, x_y_range.y_max
    x_vals = np.linspace(x_min, x_max, num_grid_points_inter)
    y_vals = np.linspace(y_min, y_max, num_grid_points_inter)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Assign colors
    intensity_interp = []
    for point in grid_points:
        simplex_index = tri_delaunay.find_simplex(point)
        # breakpoint()
        if simplex_index >= 0:  # If inside a triangle

            triangle_indices = tri_delaunay.simplices[
                simplex_index
            ]  # Indices of triangle vertices

            # breakpoint()
            vertices = tri_delaunay.points[
                triangle_indices
            ]  # Get the actual (x, y) coordinates
            v1 = vertices[0, :]
            v2 = vertices[1, :]
            v3 = vertices[2, :]

            if bVertex_based:
                a_ratio = feature_values[triangle_indices[0]]
                b_ratio = feature_values[triangle_indices[1]]
                c_ratio = feature_values[triangle_indices[2]]
            else:
                a_ratio = feature_values[simplex_index, 0]
                b_ratio = feature_values[simplex_index, 1]
                c_ratio = feature_values[simplex_index, 2]

            A1, det_A1 = create_matrix_determinant(v2, v3, point)
            A2, det_A2 = create_matrix_determinant(v3, v1, point)
            A3, det_A3 = create_matrix_determinant(v1, v2, point)

            A = det_A1 + det_A2 + det_A3

            h0 = 1 - (det_A1 / A) - (det_A2 / A)
            h1 = 1 - (det_A2 / A) - (det_A3 / A)
            h2 = 1 - (det_A3 / A) - (det_A1 / A)

            e_0 = (det_A1 / A) * h2 / (h2 + h0) + (det_A2 / A) * h1 / (h1 + h0)
            e_1 = (det_A2 / A) * h0 / (h0 + h1) + (det_A3 / A) * h2 / (h2 + h1)
            e_2 = (det_A3 / A) * h1 / (h1 + h2) + (det_A1 / A) * h0 / (h0 + h2)

            f_x = a_ratio * e_0 + b_ratio * e_1 + c_ratio * e_2

            intensity_interp.append(f_x)
        else:  # If grid outside the denaulay triangle
            intensity_interp.append(-1)  # Use 0 to indicate black color

    # Convert colors to a NumPy array
    intensity_interp = np.array(intensity_interp)

    # # Create a mask
    mask_not_valid = intensity_interp == -1

    # ___Checks_________________________
    np.any(
        (intensity_interp < 0) & (intensity_interp != -1)
    )  ##check negative value exists other than -1

    # # Set outside-triangle points to NaN
    intensity_interp[mask_not_valid] = np.nan
    # # Normalize only valid (non-NaN) values
    if np.any(~mask_not_valid):
        min_val = np.nanmin(
            intensity_interp
        )  # if we not use simple np.min(), max() it will give all nan.
        max_val = np.nanmax(intensity_interp)

        if max_val > min_val:  # Avoid division by zero
            if blog:
                intensity_interp[~mask_not_valid] = (
                    np.log(intensity_interp[~mask_not_valid]) - np.log(min_val)
                ) / (np.log(max_val) - np.log(min_val))
            else:
                intensity_interp[~mask_not_valid] = (
                    intensity_interp[~mask_not_valid] - min_val
                ) / (max_val - min_val)
        else:
            print("Warning: No variation in values, skipping normalization.")

        if bclamping:
            # Compute the 5th and 95th percentiles
            lower_bound = np.percentile(
                intensity_interp[~mask_not_valid], 1
            )  # Bottom 5% threshold

            upper_bound = np.percentile(
                intensity_interp[~mask_not_valid], 99
            )  # Top 5% threshold

            upper_bound = 1

            intensity_interp[~mask_not_valid] = np.clip(
                intensity_interp[~mask_not_valid], lower_bound, upper_bound
            )  # Clamping
            intensity_interp[~mask_not_valid] = (
                intensity_interp[~mask_not_valid] - lower_bound
            ) / (upper_bound - lower_bound)

    # # Reshape for plotting
    intensity_interp_reshape = intensity_interp.reshape(xx.shape)

    return intensity_interp_reshape, max_val, min_val


def generate_checkviz_lab_colormap(resolution=256):
    """
    Reproduces CheckViz-style 2D color map in CIELab space using three anchors.
    - X-axis = P_CCA (false neighbors)
    - Y-axis = P_NLM (tears)
    """
    cmap = np.zeros((resolution, resolution, 3), dtype=np.float32)

    for i in range(resolution):  # rows = P_NLM (tears)
        for j in range(resolution):  # cols = P_CCA (false neighbors)
            p_nlm = i / (resolution - 1)
            p_cca = j / (resolution - 1)

            # Define blend weights (based on distortion position)
            w_t = p_nlm * (1 - p_cca)  # tear only (top-left)
            w_fn = p_cca * (1 - p_nlm)  # false neighbor only (bottom-right)
            w_both = p_nlm * p_cca  # both distortions (top-right)
            w_none = (1 - p_nlm) * (1 - p_cca)  # no distortion (bottom-left)

            total = w_t + w_fn + w_both + w_none
            w_t /= total
            w_fn /= total
            w_both /= total
            w_none /= total

            # Define Lab anchors
            lab_none = np.array([100, 0, 0])  # white/light gray
            lab_t = np.array([65, -30, 20])  # green
            lab_fn = np.array([65, 30, -20])  # purple
            lab_both = np.array([30, 0, 0])  # dark

            # Weighted blend in CIELab
            lab = w_none * lab_none + w_t * lab_t + w_fn * lab_fn + w_both * lab_both

            # Convert to RGB
            rgb = cspace_convert(lab, start="CIELab", end="sRGB1")
            cmap[i, j] = np.clip(rgb, 0, 1)

    return cmap


def clipped_voronoi_polygons_2d(vor, bbox):
    

    new_regions = []
    new_vertices = []

    box = Polygon(
        [[bbox[0], bbox[2]], [bbox[1], bbox[2]], [bbox[1], bbox[3]], [bbox[0], bbox[3]]]
    )

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue  # skip infinite
        polygon = Polygon([vor.vertices[i] for i in region])
        clipped = polygon.intersection(box)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "Polygon":
            new_regions.append(np.array(clipped.exterior.coords))
        elif clipped.geom_type == "MultiPolygon":
            for poly in clipped.geoms:
                new_regions.append(np.array(poly.exterior.coords))
    return new_regions


def calculate_NLM_CCA_probability(D, low_dm_emb, sigma=None):
    """
    Calculates the Normalized Local Metric (NLM) and Class Consistency Agreement (CCA) probabilities
    for each data point in the projection space.

    Parameters:
        D (np.ndarray): High-dimensional data (original data).
        embedding (np.ndarray): 2D projection of the data (low dimensional data).
        sigma (float | None): Gaussian kernel width used for computing local neighborhoods. Default it uses distance to 5th Nearest neighbors.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized probability arrays (P_NLM, P_CCA) for all data points.
    """
    # Compute distance matrices
    D_orig = pairwise_distances(D)
    D_proj = pairwise_distances(low_dm_emb)

    D_orig = (D_orig - np.min(D_orig)) / (np.max(D_orig) - np.min(D_orig))
    D_proj = (D_proj - np.min(D_proj)) / (np.max(D_proj) - np.min(D_proj))

    if sigma is None:
        # Define sigma (scale parameter)
        sigma = np.mean(np.sort(D_orig, axis=1)[:, 5])  # distance to 5th NN

    # Define F_sigma (Heaviside step)
    def F_sigma(dist, sigma):
        return (dist < sigma).astype(float)

    # Compute local distortions for each point
    N = len(D)
    P_NLM = []
    P_CCA = []

    for i in range(N):
        # P_NLM: emphasizes small original distances
        Pn = np.sum((D_orig[i] - D_proj[i]) ** 2 * F_sigma(D_orig[i], sigma))
        P_NLM.append(Pn)

        # P_CCA: emphasizes small projection distances
        Pc = np.sum((D_orig[i] - D_proj[i]) ** 2 * F_sigma(D_proj[i], sigma))
        P_CCA.append(Pc)

    P_NLM = np.array(P_NLM)
    P_CCA = np.array(P_CCA)

    # Normalize to [0, 1]
    P_NLM_norm = Normalize(vmin=np.min(P_NLM), vmax=np.max(P_NLM))(P_NLM)
    P_CCA_norm = Normalize(vmin=np.min(P_CCA), vmax=np.max(P_CCA))(P_CCA)

    return P_NLM_norm, P_CCA_norm
    # return 1 - P_NLM_norm, 1 - P_CCA_norm  # Convert from distortion measure to quality measure to accomodate with checkviz implementation of colorcoding


##______________ZADU IMPLEMENTATION for CHECKVIz_____________________


def cielab_to_rgb_hex(L, a, b):
    # Convert CIELAB to XYZ
    def lab_to_xyz(L, a, b):
        y = (L + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        x = x**3 if x**3 > 0.008856 else (x - 16 / 116) / 7.787
        y = y**3 if y**3 > 0.008856 else (y - 16 / 116) / 7.787
        z = z**3 if z**3 > 0.008856 else (z - 16 / 116) / 7.787

        x = x * 95.047
        y = y * 100
        z = z * 108.883

        return x, y, z

    # Convert XYZ to RGB
    def xyz_to_rgb(x, y, z):
        x /= 100
        y /= 100
        z /= 100

        r = x * 3.2406 + y * -1.5372 + z * -0.4986
        g = x * -0.9689 + y * 1.8758 + z * 0.0415
        b = x * 0.0557 + y * -0.2040 + z * 1.0570

        def convert(color):
            color = np.clip(color, 0, 1)
            color = np.where(
                color <= 0.0031308,
                12.92 * color,
                1.055 * np.power(color, 1 / 2.4) - 0.055,
            )
            return np.round(color * 255).astype(int)

        r, g, b = convert(r), convert(g), convert(b)
        return r, g, b

    x, y, z = lab_to_xyz(L, a, b)
    r, g, b = xyz_to_rgb(x, y, z)
    return f"#{r:02x}{g:02x}{b:02x}"


##____ above is for CheckViz colorMap________________________________________________________
def checkviz_cmap(dist_false, dist_missing):
    """
    Maps two probability scores (CCA and NLM) to a unique color using the CheckViz colormap.

    Parameters:
        pcca (float): Probability score for CCA. (Missing neighbors = Tear)
        pnml (float): Probability score for NLM. (False neighbors)

    Returns:
        str: Hex color code representing the mapped color.
    """

    cScale = 1.3

    # dist_false = 1 - dist_false
    # dist_missing = 1 - dist_missing

    powScale = lambda x: x**1.5145
    aScale = lambda x: 30 * cScale * x
    bScale = lambda x: 20 * cScale * x

    lab = [
        powScale(1 - (dist_false + dist_missing) / 2) * 100,
        aScale(dist_false - dist_missing),
        bScale(dist_missing - dist_false),
    ]

    ## change the cielab color to rgb that can be used by matplotlib
    color = cielab_to_rgb_hex(*lab)
    return color


def hex_to_rgb_normalized_or_nan(hex_color):
    """
    Converts a hex color string to a normalized RGB tuple (range [0, 1]).
    Returns NaNs for invalid colors.

    Parameters:
        hex_color (str): Hex color string (e.g., '#AABBCC').

    Returns:
        Tuple[float, float, float]: Normalized RGB values or NaNs.
    """

    if isinstance(hex_color, str) and hex_color.startswith("#"):
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    else:
        return (np.nan, np.nan, np.nan)






def get_reducer(method, perplexity):
    if method == "tsne":
        reducer = manifold.TSNE(
            n_components=2, perplexity=perplexity, init="random", random_state=cfg.GLOBAL_SEED
        )

    elif method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=perplexity,
            min_dist=0.1,
            init="random",
            random_state=cfg.GLOBAL_SEED,
        )

    elif method == "pca":
        reducer = PCA(n_components=2, random_state=cfg.GLOBAL_SEED)


    return reducer


def inter_intra_cluster_pairwise_distance_optimised(data, labels,
                                          metric='euclidean',
                                          norm_distance=False):

    unique_clusters = np.sort(np.unique(labels))
    num_clusters = len(unique_clusters)

    # Precompute cluster subsets
    cluster_points = [data[labels == c] for c in unique_clusters]
    cluster_sizes = [len(cp) for cp in cluster_points]

    # Preallocate full matrix (float32 = half memory)
    totalN = np.sum(cluster_sizes)
    full_pairwise = np.zeros((totalN, totalN), dtype=np.float32)

    # Preallocate mean cluster distance matrix
    mean_cluster = np.zeros((num_clusters, num_clusters), dtype=np.float32)

    # Compute block boundaries
    boundaries = np.cumsum([0] + cluster_sizes)

    # Compute pairwise blocks efficiently
    for i in range(num_clusters):
        Xi = cluster_points[i]

        for j in range(i, num_clusters):
            Xj = cluster_points[j]

            # Compute distance block
            block = pairwise_distances(Xi, Xj, metric=metric).astype(np.float32)

            # Insert into full matrix
            r0, r1 = boundaries[i], boundaries[i+1]
            c0, c1 = boundaries[j], boundaries[j+1]

            full_pairwise[r0:r1, c0:c1] = block

            if i != j:
                full_pairwise[c0:c1, r0:r1] = block.T

            # Compute mean
            mean_val = float(np.mean(block))
            mean_cluster[i, j] = mean_val
            mean_cluster[j, i] = mean_val

    # Normalize *in place* (no new matrix created)
    if norm_distance:
        min_val = float(full_pairwise.min())
        max_val = float(full_pairwise.max())
        full_pairwise -= min_val
        full_pairwise /= (max_val - min_val + 1e-12)

        min_m = float(mean_cluster.min())
        max_m = float(mean_cluster.max())
        mean_cluster -= min_m
        mean_cluster /= (max_m - min_m + 1e-12)

    return full_pairwise, mean_cluster


####____________tool_____________________________________________________



def rasterize_lines(line_segments, colors, image_size, blend_method="max"):
    H, W = image_size

    img = np.zeros((H, W, 3), dtype=np.float32)  # Black background
    count = np.zeros((H, W), dtype=np.float32)  # To track overlapping lines
    visited = np.zeros((H, W), dtype=bool)

    for i, (p1, p2) in enumerate(line_segments):
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)

        # Get pixel coordinates of the line
        rr, cc = rasterize_line(y1, x1, y2, x2)

        # Ensure indices are within bounds
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid], cc[valid]

        visited[rr, cc] = True

        # Extract current pixel values
        current_pixels = img[rr, cc]
        new_color = colors[i][:3]

        if blend_method == "average":
            img[rr, cc] += new_color
            count[rr, cc] += 1

        elif blend_method == "max":
            mask = np.linalg.norm(new_color) >= np.linalg.norm(current_pixels, axis=1)
            img[rr[mask], cc[mask]] = new_color  # Apply new color only where needed

        elif blend_method == "min":
            new_color_norm = np.linalg.norm(new_color)
            current_norms = np.linalg.norm(current_pixels, axis=1)

            # Find where the new color should replace the current pixel
            mask = (current_norms > new_color_norm) | np.all(
                current_pixels == [0, 0, 0], axis=1
            )
            img[rr[mask], cc[mask]] = new_color  # Apply the minimum color efficiently

    #
    # Normalize for "average" blending
    if blend_method == "average":
        mask = count > 0
        img[mask] /= count[mask][:, None]
        img[~mask] = 1  # Set not used pixels to white background

    if blend_method in ["min", "max"]:
        # breakpoint()
        img[~visited] = 1  # Set not used pixels to white background

    # Normalize image contrast
    valid_pixels = count > 0
    if np.any(visited):
        min_val = np.min(img[visited])
        max_val = np.max(img[visited])

        if max_val > min_val:
            img[visited] = (img[visited] - min_val) / (max_val - min_val)

    return img


##-------- START: Computation Cost-------------------------


def run_and_measure(func, *args, repeats=5, **kwargs):
    runtimes = []
    output = None

    for i in range(repeats):
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        end = time.perf_counter()

        runtimes.append(end - start)

        if i == 0:
            output = result

    return output, np.mean(runtimes), np.std(runtimes)

##-------- END: Computation Cost-------------------------

###---------START: CUSTOM ColorMap-------------------------------
def get_cool_glasbey(n_colors=20,
                     max_luminance=0.85,
                     min_luminance=0.15,
                     white_threshold=0.95):

    # Convert HEX → RGB
    colors = np.array([to_rgb(c) for c in cc.glasbey_light])

    # Perceived luminance (sRGB standard)
    luminance = (
        0.2126 * colors[:, 0] +
        0.7152 * colors[:, 1] +
        0.0722 * colors[:, 2]
    )

    # Remove near-white colors
    near_white = np.all(colors > white_threshold, axis=1)

    # Remove near-black colors
    near_black = np.all(colors < 0.05, axis=1)

    # Remove warm (red-dominant) colors
    red_dominant = colors[:, 0] > 0.6

    # Remove too bright or too dark
    luminance_mask = (luminance < max_luminance) & (luminance > min_luminance)

    # Combine all filters
    mask = (
        luminance_mask &
        (~near_white) &
        (~near_black) &
        (~red_dominant)
    )

    filtered = colors[mask]

    if len(filtered) < n_colors:
        raise ValueError(
            f"Only {len(filtered)} usable colors after filtering. "
            "Relax thresholds."
        )

    return ListedColormap(filtered[:n_colors])
###---------END: CUSTOM ColorMap-------------------------------

###-----------START: LOAD numpyarray------------------------------
from pathlib import Path
import numpy as np


def load_numpy_array(folder, name, dataset, ext=None, allow_pickle=False):
    base = Path(folder) / f"{name}_{dataset}"

    npy_path = base.with_suffix(".npy")
    npz_path = base.with_suffix(".npz")

    # If extension explicitly provided
    if ext:
        path = base.with_suffix(f".{ext}")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return np.load(path, allow_pickle=allow_pickle)

    # Automatic detection
    exists_npy = npy_path.exists()
    exists_npz = npz_path.exists()

    if exists_npy and exists_npz:
        raise ValueError(
            f"Both {npy_path.name} and {npz_path.name} exist. "
            "Specify ext='npy' or ext='npz'."
        )

    if exists_npy:
        return np.load(npy_path, allow_pickle=allow_pickle)

    if exists_npz:
        return np.load(npz_path, allow_pickle=allow_pickle)

    raise FileNotFoundError(
        f"No file found: {npy_path.name} or {npz_path.name}"
    )
###-----------END: LOAD numpyarray------------------------------




def cubic_centers_sizes(dataset):

    # centers = np.array([
    #     (0, 0, 0),
    #     (10, 0, 0),
    #     (5, 8, 0),
    #     (5, 4, 7)
    # ], dtype=float)

    # cube_sizes = None

    cluster_spacing = 10.0
    centers = np.array(
                [
                    [cluster_spacing, cluster_spacing, cluster_spacing],
                    [-cluster_spacing, -cluster_spacing, cluster_spacing],
                    [-cluster_spacing, cluster_spacing, -cluster_spacing],
                    [cluster_spacing, -cluster_spacing, -cluster_spacing],
                ]
            ) / np.sqrt(3)

    cube_sizes = None
    cube_centers = None


    if dataset =='cubic':
        cube_sizes=(2.0, 5.0, 10.0)
        # c_s = 3.0  # 5.0 is good
        # cube_sizes=(c_s, c_s, c_s)
        # cube_centers = ( (0, 0, 0),  (8, 8, 8), (16, 16, 16))    ## user study 
        cube_centers = ( (0, 0, 0),  (10, 10, 10), (30, 30, 30))    ## paper teaser
    elif dataset =='cubic_2_clusters':
        # cube_sizes=(10.0, 5.0, 1.0)
        cube_sizes=(10, 2)
        cube_centers = ( (0, 0, 0),  (16, 16, 16))    ## orienwted on main diogonal

    elif dataset in ['cube_diff_size_diff_dist']:
        cube_sizes=(15.0, 4.0, 9.0, 1.0) # used inpaper
        # cube_sizes=(25.0, 15.0, 9.0, 4.0)
        centers[0] *= 5
        centers[1]  *= 1
        centers[2]  *= 2
        cube_centers = centers

    elif dataset == 'cube_same_size_same_dist':
        c_s = 2.0  # 2.0 is used in paper
        # cube_sizes=(c_s, c_s, c_s)
        cube_sizes=(c_s, c_s, c_s, c_s)
        # s = 10
        # cube_centers = ( (0, 0, 0),  (s, 0, 0), (s/2, s * np.sqrt(3)/2.0, 0))
        cube_centers = centers*2

    elif dataset == 'cube_diff_size_same_dist':
        # cube_sizes=(10.0, 5.0, 1.0) 
        # cube_sizes=(30.0, 15.0, 5.0, 1.0) #used in paper
        cube_sizes=(30.0, 15.0, 8.0, 4.0) 
        # s = 10
        # cube_centers = ( (0, 0, 0),  (s, 0, 0), (s/2, s * np.sqrt(3)/2.0, 0))
        # centers[2]  *= 2
        cube_centers = centers*6

    elif dataset == 'cube_same_size_diff_dist':
        # cube_sizes=(10.0, 5.0, 1.0)
        c_s = 3.0  # 5.0 is good
        cube_sizes=(c_s, c_s, c_s, c_s)
        # s = 10
        # cube_centers = ( (0, 0, 0),  (s, 0, 0), (s/2, s * np.sqrt(3)/2.0, 0))
        centers[3] *= 3
        centers[1]  *= 1
        cube_centers = centers

    return cube_sizes, cube_centers

####_____________ Sensitivity analysis________________________________________________________

# ============================================================
# 1. Perturb data
# ============================================================
def perturb_dataset(X, noise_ratio, random_state=42):
    """
    Add Gaussian noise proportional to each feature std.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)

    noise_ratio : float
        e.g. 0.01 means 1% of std

    Returns
    -------
    X_perturbed
    """

    rng = np.random.default_rng(random_state)

    feature_std = np.std(X, axis=0)

    noise = rng.normal(
        loc=0,
        scale=noise_ratio * feature_std,
        size=X.shape
    )

    return X + noise

def compare_edge_sets(reference_edges, perturbed_edges):
    
    def edges_to_set(edges):
        return set(map(tuple, edges))
    
    reference_edges = edges_to_set(reference_edges)
    perturbed_edges = edges_to_set(perturbed_edges)

    preserved = reference_edges.intersection(
        perturbed_edges
    )

    lost = reference_edges.difference(
        perturbed_edges
    )

    new = perturbed_edges.difference(
        reference_edges
    )

    n_reference = len(reference_edges)

    preserved_ratio = len(preserved) / n_reference

    lost_ratio = len(lost) / n_reference

    new_ratio = len(new) / n_reference

    union_size = len(
        reference_edges.union(perturbed_edges)
    )

    jaccard = len(preserved) / union_size

    return {
        "preserved_edges": preserved,
        "lost_edges": lost,
        "new_edges": new,

        "preserved_ratio": preserved_ratio,
        "lost_ratio": lost_ratio,
        "new_ratio": new_ratio,

        "jaccard_similarity": jaccard
    }

def extract_delaunay_edges_2d(tri):
    """
    Compute Delaunay triangulation and extract unique edges.

    Parameters
    ----------
    X : (N,2) array
        2D points

    Returns
    -------
    edges : (E,2) array
        Unique edges (node index pairs)
    triangles : (T,3) array
        Triangle indices (optional useful output)
    """

    # --- triangulation ---
    # tri = Delaunay(X)
    triangles = tri.simplices  # (T,3)

    # --- extract edges from triangles ---
    e1 = triangles[:, [0, 1]]
    e2 = triangles[:, [1, 2]]
    e3 = triangles[:, [2, 0]]

    edges = np.vstack((e1, e2, e3))  # (3T,2)

    # --- remove duplicates (order-independent) ---
    edges = np.sort(edges, axis=1)          # ensure (i,j) == (j,i)
    edges = np.unique(edges, axis=0)        # keep unique edges

    return edges, triangles
