# import itertools
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colormaps

import numpy as np
import plotly.graph_objects as go

from scipy.spatial import Voronoi


import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from matplotlib.colors import to_rgb


from .utility import *

colors = [
    "#FF0000",
    "#00FF00",
    "#FF00FF",
    "#FFFF00",
    "#00FFFF",
    "#0000FF",
    "#000000",
    "#FFA500",
    "#8000FF",
    "#FF1493",
]


data_point_size = 30
figsize = (10,10)
save_format = "png"
dpi = 300
# colormap = 'plasma'



def plot_pairwise_cluster_distance_v2(
    distance_matrix,
    mean_cluster_distance,
    label,
    output_path=None,
    border_thickness=15,
    fontsize = 20,
    figsize=figsize,
    colormap="plasma",       
    save_format="png",
    filename = "high_dm",
    dpi=300,
    use_single_color = False,
    show_mean_distances = True,
):
    """
    Plots pairwise cluster distances with dynamic colors based on labels.

    Parameters:
        distance_matrix (np.ndarray): Pairwise distance matrix.
        mean_cluster_distance (np.ndarray): Mean distances for each cluster pair.
        label (np.ndarray): Cluster labels for each sample.
        colors (list or None): Optional list of colors. If None, generated from colormap.
        output_path (str or None): Directory to save the plot. If None, shows plot.
        border_thickness (int): Thickness of the cluster border lines.
        figsize (tuple): Figure size.
        colormap (str or Colormap): Colormap name or object for generating colors.
        save_format (str): Image format to save as.
    """

    unique_clusters = np.sort(np.unique(label))

    # ----- Generate colors if not provided -----
    if isinstance(colormap, str):
        cmap = colormaps.get_cmap(colormap)
    elif isinstance(colormap, mcolors.Colormap):
        cmap = colormap
    else:
        cmap = colormaps.get_cmap("tab10")  # Fallback


    num_clusters = len(unique_clusters)
    colors = [cmap(i / max(1, num_clusters - 1)) for i in range(num_clusters)]

    # Compute cluster boundaries
    num_points_per_cluster = [np.sum(label == i) for i in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)

    plt.rcParams["figure.dpi"] = dpi      # display resolution
    plt.rcParams["savefig.dpi"] = dpi     # saved file resolution

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    ax.set_aspect(1)
    ax.axis("equal")
    plt.axis("off")
    im = ax.imshow(distance_matrix, cmap="hot", rasterized=True)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    # Draw cluster borders
    for i in range(len(unique_clusters)):
        pad_border_corner = 0
        start = cumulative_positions[i] - pad_border_corner
        end = cumulative_positions[i + 1] - pad_border_corner

        

        if not use_single_color:
            color = colors[i]
            # Left border
            ax.plot(
                [-0.5, -0.5],
                [start - 0.5, end - 0.5],
                color=color,
                linewidth=border_thickness,
                solid_capstyle="butt",
                clip_on=False,
            )

            # Top border
            ax.plot(
                [start - 0.5, end - 0.5],
                [-0.5, -0.5],
                color=color,
                linewidth=border_thickness,
                solid_capstyle="butt",
                clip_on=False,
            )
        else:
            color = colors[i]
            # Left border
            ax.plot(
                [-0.5, -0.5],
                [start - 0.5, end - 0.5],
                color="#ffffff",
                linewidth=border_thickness,
                solid_capstyle="butt",
                clip_on=False,
            )

            # Top border
            ax.plot(
                [start - 0.5, end - 0.5],
                [-0.5, -0.5],
                color="#ffffff",
                linewidth=border_thickness,
                solid_capstyle="butt",
                clip_on=False,
            )

    if show_mean_distances:
        # Annotate mean distances
        for i, start_i in enumerate(cumulative_positions[:-1]):
            for j, start_j in enumerate(cumulative_positions[:-1]):
                mean_distance = mean_cluster_distance[i, j]
                center_x = (start_j + cumulative_positions[j + 1]) / 2
                center_y = (start_i + cumulative_positions[i + 1]) / 2
                # Choose color depending on diagonal vs. off-diagonal
                text_color = "white" if i == j else "black"
                ax.text(
                    center_x,
                    center_y,
                    f"{mean_distance:.2f}",
                    color=text_color,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    bbox=dict(
                        # boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.0
                        boxstyle="round", facecolor="none", edgecolor="0.3"
                    ),
                )

    # Save or show
    if output_path:
        output_file = os.path.join(output_path, f"pairwise_distance_{filename}.{save_format}")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_pairwise_cluster_distance_v2_custom_color(
    distance_matrix,
    mean_cluster_distance,
    label,
    color_list=None,   # ONLY for borders

    output_path=None,
    border_thickness=15,
    fontsize=20,
    figsize=(10, 10),
    save_format="png",
    filename="high_dm",
    dpi=300,
    show_mean_distances=True,
):
    """
    Pairwise cluster distance plot with:
    - Heatmap (fixed: 'hot')
    - Custom color list ONLY for outer borders
    """

    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.distiller.res'] = 6000  # high-quality raster parts

    # ---------------- DPI SETTINGS ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi


    # ---------------- CLUSTERS ----------------
    unique_clusters = np.sort(np.unique(label))
    num_clusters = len(unique_clusters)

    if color_list is None:
        raise ValueError("You must provide color_list for cluster borders.")

    # COLOR MAPPING (ORDER PRESERVED)
    cluster_color_map = {
        cls: color_list[i % len(color_list)]
        for i, cls in enumerate(unique_clusters)
    }

    # ---------------- CLUSTER SIZES ----------------
    num_points_per_cluster = [np.sum(label == c) for c in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)

    # ---------------- FIGURE ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # ---------------- HEATMAP ----------------
    im = ax.imshow(distance_matrix, cmap="hot", rasterized=True)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---------------- BORDERS ----------------
    for i, cls in enumerate(unique_clusters):

        start = cumulative_positions[i]
        end = cumulative_positions[i + 1]

        color = cluster_color_map[cls]

        # ---- LEFT BORDER ----
        ax.plot(
            [-0.5, -0.5],
            [start - 0.5, end - 0.5],
            color=color,
            linewidth=border_thickness,
            solid_capstyle="butt",
            clip_on=False,
        )

        # ---- TOP BORDER ----
        ax.plot(
            [start - 0.5, end - 0.5],
            [-0.5, -0.5],
            color=color,
            linewidth=border_thickness,
            solid_capstyle="butt",
            clip_on=False,
        )

    # ---------------- MEAN DISTANCE TEXT ----------------
    if show_mean_distances:

        for i, start_i in enumerate(cumulative_positions[:-1]):
            for j, start_j in enumerate(cumulative_positions[:-1]):

                mean_distance = mean_cluster_distance[i, j]

                center_x = (start_j + cumulative_positions[j + 1]) / 2
                center_y = (start_i + cumulative_positions[i + 1]) / 2

                text_color = "white" if i == j else "black"

                ax.text(
                    center_x,
                    center_y,
                    f"{mean_distance:.2f}",
                    color=text_color,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground='black')
                    ],
                    bbox=dict(
                        boxstyle="round",
                        facecolor="none",
                        edgecolor="0.3"
                    ),
                )

    # ---------------- SAVE / SHOW ----------------
    if output_path:
        output_file = os.path.join(
            output_path,
            f"pairwise_distance_{filename}"
        )
        plt.savefig(f"{output_file}.{save_format}", bbox_inches="tight", pad_inches=0)
        # plt.savefig(f"{output_file}.eps", format="eps", dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()



def plot_delaunay_triangulation(
    tri_delaunay,
    embedding,
    class_label=None,
    title="Delaunay Triangulation of t-SNE Output",
    colormap ='plasma',
    tri_color="blue",
    alpha=0.7,
    figsize=(10,10),
    data_point_size = 30,
    save_format = 'png',
    dpi = 300,
    output_path=None,
    filename = None
):
    """
    Plots the Delaunay triangulation of a 2D embedding and optionally saves the figure.

    Parameters:
        tri_delaunay (scipy.spatial.Delaunay): Delaunay triangulation object containing simplices.
        embedding (np.ndarray): 2D array of shape (n_samples, 2) representing the projected points.
        class_label (np.ndarray, optional): Class labels for coloring points. If None, all points are gray.
        title (str, optional): Title of the plot. Default is "Delaunay Triangulation of t-SNE Output".
        colormap (str or matplotlib.colors.Colormap, optional): Colormap for different classes. Default is `plasma`.
        tri_color (str, optional): Color of the triangulation lines. Default is 'blue'.
        alpha (float, optional): Transparency level of the triangulation lines. Default is 0.7.
        figsize (tuple, optional): Figure size (width, height). Default is `(10,10)`.
        data_point_size (float, optional): Size of the scatter plot markers. Default is `30`.
        save_format (str, optional): File format to save the figure ('png', 'pdf', etc.). Default is `png`.
        dpi (int, optional): Resolution of the saved figure. Default is `600`.
        output_path (str, optional): Path to save the figure. If None, the plot is displayed instead.

    Notes:
        - If `class_label` is provided, each unique class is colored differently.
        - Axes ticks are removed, but spines are kept visible.
        - The plot uses equal aspect ratio and hides the axis for cleaner visualization.
        - If `output_path` is not specified, the plot is shown but not saved.

    Returns:
        None
    """

    # Create figure
    plt.figure(figsize=figsize, constrained_layout=True)

    ax = plt.gca()
    # Plot triangulation
    ax.triplot(
        embedding[:, 0],
        embedding[:, 1],
        tri_delaunay.simplices,
        color=tri_color,
        alpha=alpha,
        linewidth=0.8,
        linestyle="-",
    )

    n_gauss = len(np.unique(class_label))
    # Plot scatter points
    if class_label is not None and n_gauss is not None:
        # Generate color palette
        # colmap = colormaps.get_cmap(colormap, n_gauss)
        colmap = colormaps.get_cmap(colormap).resampled(n_gauss)
        colors = [colmap(i) for i in range(n_gauss)]

        for i in range(n_gauss):
            ax.scatter(
                embedding[class_label == i, 0],
                embedding[class_label == i, 1],
                color=colors[i],
                zorder=3,
                edgecolor="k",
                s=data_point_size,
            )
    else:
        # Fallback: simple scatterplot without labels or colors
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            color="gray",
            edgecolor="black",
            s=data_point_size,
        )

    ax.axis("equal")
    plt.axis("off")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Save or show
    if output_path:
        output_file = os.path.join(output_path, f"delaunay_triangulation_{filename}.{save_format}")
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()






def filter_k_edges_per_cluster(edges, edge_lengths, class_label, k, mode="shortest"):
    """
    Keep k edges per cluster based on length.

    Parameters
    ----------
    edges : (E, 2)
    edge_lengths : (E,)
    class_label : (N,)
    k : int
    mode : 'shortest' or 'longest'

    Returns
    -------
    filtered_edges, filtered_lengths
    """

    edges = np.asarray(edges)
    edge_lengths = np.asarray(edge_lengths)

    cluster_to_edges = {}

    # ---- assign edges to clusters ----
    for i, (idx1, idx2) in enumerate(edges):
        c1 = class_label[idx1]
        c2 = class_label[idx2]

        for c in (c1, c2):
            if c not in cluster_to_edges:
                cluster_to_edges[c] = []
            cluster_to_edges[c].append((i, edge_lengths[i]))

    selected_indices = set()

    # ---- selection per cluster ----
    for c, edge_list in cluster_to_edges.items():

        if mode == "shortest":
            edge_list = sorted(edge_list, key=lambda x: x[1])  # ascending
        elif mode == "longest":
            edge_list = sorted(edge_list, key=lambda x: x[1], reverse=True)  # descending
        else:
            raise ValueError("mode must be 'shortest' or 'longest'")

        for idx, _ in edge_list[:k]:
            selected_indices.add(idx)

    selected_indices = sorted(selected_indices)

    return edges[selected_indices], edge_lengths[selected_indices]



def build_cluster_color_map(class_label, colormap):
    """
    Returns:
        cluster_color_map: dict {label -> color}
    """

    unique_classes = np.unique(class_label)
    unique_classes = unique_classes[unique_classes != -1]
    unique_classes = np.sort(unique_classes)

    # ---------- CASE 1: USER PASSED LIST ----------
    if isinstance(colormap, list):

        n_colors = len(colormap)

        cluster_color_map = {
            cls: colormap[i % n_colors]
            for i, cls in enumerate(unique_classes)
        }

        return cluster_color_map

    # ---------- CASE 2: USER PASSED STRING OR CMAP ----------
    else:
        cmap = colormaps.get_cmap(colormap) if isinstance(colormap, str) else colormap

        # If discrete cmap (ListedColormap)
        if hasattr(cmap, "N"):
            n_colors = cmap.N

            cluster_color_map = {
                cls: cmap(i % n_colors)
                for i, cls in enumerate(unique_classes)
            }

        else:
            # fallback (continuous cmap)
            cluster_color_map = {
                cls: cmap(i / max(1, len(unique_classes)-1))
                for i, cls in enumerate(unique_classes)
            }

        return cluster_color_map


def plot_analysis(
    embedding,
    class_label=None,
    interpolations=None,
    edges=None,
    edge_lengths=None,
    show_all_edges=True,
    k_edges_per_cluster=1,
    edge_selection_mode = "shortest",

    edge_linewidth=1.0,
    shrink_factor_edge_points = 0.7,

    xy_range=None,
    bscatter_plot=False,
    background_color="white",
    output_path=None,
    figsize=(10,10),
    filename='plot',
    colormap='plasma',
    save_format='png',
    dpi=300,
    data_point_size=30,
):

    if xy_range is None:
        xy_range = x_y_range(embedding)

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    base_size = data_point_size
    # marker_size = base_size * shrink_factor_edge_points
    # edge_width = np.sqrt(base_size) * (1 - shrink_factor_edge_points) 

    marker_size = data_point_size * (shrink_factor_edge_points ** 2)
    edge_width = np.sqrt(data_point_size) * (1 - shrink_factor_edge_points)

    # ---------------- HEATMAP ----------------
    # heatmap = ax.imshow(
    #     interpolations,
    #     extent=(xy_range.x_min, xy_range.x_max, xy_range.y_min, xy_range.y_max),
    #     origin="lower",
    #     cmap="hot",
    #     alpha=1.0,
    #     interpolation="none",   # 
    #     resample=False
    # )
    # heatmap.cmap.set_bad(background_color)

    x = np.linspace(xy_range.x_min, xy_range.x_max, interpolations.shape[1])
    y = np.linspace(xy_range.y_min, xy_range.y_max, interpolations.shape[0])

    X, Y = np.meshgrid(x, y)

    heatmap = ax.pcolormesh(
        X, Y,
        interpolations,
        cmap="hot",
        shading='nearest'   
    )
    heatmap.cmap.set_bad(background_color)

    # ---------------- SCATTER ----------------
    cluster_color_map = {}

    if bscatter_plot:

        if isinstance(colormap, str):
            cmap = colormaps.get_cmap(colormap)

        else:
            cmap = colormap

        if class_label is not None:

            unique_classes = np.unique(class_label)
            unique_classes = unique_classes[unique_classes != -1]
            num_classes = len(unique_classes)

            # CREATE CONSISTENT COLOR MAP
            
            cluster_color_map = build_cluster_color_map(class_label, colormap)
            for ind, cls in enumerate(unique_classes):
                mask = class_label == cls
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    color=colors[ind],
                    edgecolor=colors[ind],
                    s=data_point_size,
                    linewidths=0.75,
                    zorder=5,
                    facecolors='none'
                )

        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                color=cmap(0.5),
                edgecolor="white",
                s=data_point_size,
                zorder=5,
                facecolors='none',
                alpha=0.5
            )
    
    # ---- PLOT NOISE POINTS (GRAY) ----
    if class_label is not None:
        noise_mask = class_label == -1

        if np.any(noise_mask):
            ax.scatter(
                embedding[noise_mask, 0],
                embedding[noise_mask, 1],
                color="gray",
                edgecolor="gray",
                s=data_point_size,
                linewidths=0.75,
                zorder=1,              # slightly below clusters
                facecolors='none',
                # alpha=0.5
            )

    # ---------------- FALLBACK COLOR MAP ----------------
    if class_label is not None and len(cluster_color_map) == 0:
        unique_classes = np.unique(class_label)
        unique_classes = unique_classes[unique_classes != -1]

        cluster_color_map = build_cluster_color_map(class_label, colormap)

    # ---------------- EDGE OVERLAY ----------------
    if edges is not None and edge_lengths is not None:

        edges = np.asarray(edges)
        edge_lengths = np.asarray(edge_lengths)

        # ---- APPLY FILTER ----
        if (not show_all_edges) and (class_label is not None):
            
            edges, edge_lengths = filter_k_edges_per_cluster(
                edges, edge_lengths, class_label, k_edges_per_cluster,
                mode=edge_selection_mode
            )

        edge_cmap = heatmap.cmap
        norm = heatmap.norm


        inner_size = 50
        edge_ring_width = 6.5
        outer_ring_width = 7.5

        for i in range(len(edges)):
            idx1, idx2 = edges[i]

            x1, y1 = embedding[idx1]
            x2, y2 = embedding[idx2]

            edge_color = edge_cmap(norm(edge_lengths[i]))

            # ---- WHITE OUTLINE (draw first, thicker) ----
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white",
                linewidth=edge_width * 1.5,   # outline thickness
                zorder=3
            )

            # ---- ACTUAL EDGE LINE (on top) ----
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edge_color,
                linewidth=edge_width,
                zorder=3
            )


            # ---- GET CLUSTER COLORS ----
            if class_label is not None:
                c1 = cluster_color_map[class_label[idx1]]
                c2 = cluster_color_map[class_label[idx2]]
            else:
                c1 = c2 = "white"


            # ---------- POINT 1 ---------- 
            # Outer white ring
            ax.scatter(
                x1, y1,
                s=inner_size,
                facecolors='none',
                edgecolors='white',
                linewidths=outer_ring_width,
                zorder=28
            )

            # Middle colored ring (edge length color)
            ax.scatter(
                x1, y1,
                s=inner_size,
                facecolors='none',
                edgecolors=edge_color,
                linewidths=edge_ring_width,
                zorder=29
            )

            # Inner filled circle (cluster color)
            ax.scatter(
                x1, y1,
                s=inner_size,
                facecolors=[c1],
                edgecolors='none',
                zorder=30
            )

            # ---------- POINT 2 ----------
            ax.scatter(
                x2, y2,
                s=inner_size,
                facecolors='none',
                edgecolors='white',
                linewidths=outer_ring_width,
                zorder=28
            )

            ax.scatter(
                x2, y2,
                s=inner_size,
                facecolors='none',
                edgecolors=edge_color,
                linewidths=edge_ring_width,
                zorder=29
            )

            ax.scatter(
                x2, y2,
                s=inner_size,
                facecolors=[c2],
                edgecolors='none',
                zorder=30
            )

    # ---------------- FINAL STYLING ----------------
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)

    # ax.set_aspect('auto')
    ax.set_aspect('equal')
    plt.axis("off")

    # ---------------- SAVE / SHOW ----------------
    if output_path:
        # file_path = os.path.join(output_path, f"{filename}.{save_format}")
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        # plt.savefig(file_path, bbox_inches="tight")
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()



def plot_analysis_custom_color(
    embedding,
    class_label=None,
    interpolations=None,
    edges=None,
    edge_lengths=None,
    color_list=None,   # ONLY color source

    show_all_edges=True,
    k_edges_per_cluster=1,
    edge_selection_mode="shortest",

    xy_range=None,
    bscatter_plot=False,
    background_color="white",

    output_path=None,
    figsize=(10, 10),
    filename='plot',
    save_format='png',
    dpi=300,
    data_point_size=30,
    linewidths = 0.20
):  
    
    
    

    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.distiller.res'] = 6000  # high-quality raster parts

    # ---------------- DPI SETTINGS ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    # ---------------- RANGE ----------------
    if xy_range is None:
        xy_range = x_y_range(embedding)

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0, 0, 1, 1])

    # ---------------- HEATMAP ----------------
    
    heatmap = ax.imshow(
        interpolations,
        extent=(xy_range.x_min, xy_range.x_max, xy_range.y_min, xy_range.y_max),
        origin="lower",
        cmap="hot",
        alpha=1.0,
        interpolation="none",   # 🔥 key
        resample=False
    )
    heatmap.cmap.set_bad(background_color)

    # ---------------- CLUSTER COLORS ----------------
    cluster_color_map = {}

    if class_label is not None and color_list is not None:
        unique_classes = np.unique(class_label)
        unique_classes = unique_classes[unique_classes != -1]

        # ORDER COLOR ASSIGNMENT
        cluster_color_map = {
            cls: color_list[i % len(color_list)]
            for i, cls in enumerate(unique_classes)
        }

    # ---------------- SCATTER ----------------
    if bscatter_plot and class_label is not None:

        for cls, color in cluster_color_map.items():
            mask = class_label == cls

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                facecolors='none',
                edgecolors=color,
                s=data_point_size,
                linewidths=linewidths,
                zorder=5
            )

        # ---- NOISE ----
        noise_mask = class_label == -1
        if np.any(noise_mask):
            ax.scatter(
                embedding[noise_mask, 0],
                embedding[noise_mask, 1],
                facecolors='none',
                edgecolors="#453D3DF7",
                s=data_point_size,
                linewidths=0.75,
                zorder=3
            )

    # ---------------- EDGE OVERLAY ----------------
    if edges is not None and edge_lengths is not None and heatmap is not None:

        edges = np.asarray(edges)
        edge_lengths = np.asarray(edge_lengths)

        if (not show_all_edges) and (class_label is not None):
            edges, edge_lengths = filter_k_edges_per_cluster(
                edges,
                edge_lengths,
                class_label,
                k_edges_per_cluster,
                mode=edge_selection_mode
            )

        edge_cmap = heatmap.cmap
        norm = heatmap.norm

        # ---- USER CONTROL ----
        inner_size = 50
        edge_ring_width = 6.5
        outer_ring_width = 7.5
        line_width = 1.2

        for i in range(len(edges)):
            idx1, idx2 = edges[i]

            x1, y1 = embedding[idx1]
            x2, y2 = embedding[idx2]

            edge_color = edge_cmap(norm(edge_lengths[i]))

            # ---- EDGE LINE ----
            ax.plot([x1, x2], [y1, y2], color="white",
                    linewidth=line_width * 1.5, zorder=3)

            ax.plot([x1, x2], [y1, y2], color=edge_color,
                    linewidth=line_width, zorder=3)

            # ---- CLUSTER COLORS ----
            if class_label is not None:
                c1 = cluster_color_map.get(class_label[idx1], "white")
                c2 = cluster_color_map.get(class_label[idx2], "white")
            else:
                c1 = c2 = "white"

            # ---------- POINT 1 ----------
            ax.scatter(x1, y1, s=inner_size,
                       facecolors='none', edgecolors='white',
                       linewidths=outer_ring_width, zorder=28)

            ax.scatter(x1, y1, s=inner_size,
                       facecolors='none', edgecolors=edge_color,
                       linewidths=edge_ring_width, zorder=29)

            ax.scatter(x1, y1, s=inner_size,
                       facecolors=c1, edgecolors='none', zorder=30)

            # ---------- POINT 2 ----------
            ax.scatter(x2, y2, s=inner_size,
                       facecolors='none', edgecolors='white',
                       linewidths=outer_ring_width, zorder=28)

            ax.scatter(x2, y2, s=inner_size,
                       facecolors='none', edgecolors=edge_color,
                       linewidths=edge_ring_width, zorder=29)

            ax.scatter(x2, y2, s=inner_size,
                       facecolors=c2, edgecolors='none', zorder=30)

    # ---------------- FINAL STYLE ----------------
    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)
    ax.set_aspect('equal')
    ax.axis("off")

    # ---------------- SAVE / SHOW ----------------
    if output_path:
        file_path = os.path.join(output_path, f"{filename}")
        
        plt.savefig(f"{file_path}.{save_format}")
        # plt.savefig(f"{file_path}.eps", format="eps", dpi=dpi)
        plt.show()
        plt.close()
    else:
        plt.show()



def plot_projection(
    embedding,
    class_label=None,
    background_color="white",
    output_path=None,
    figsize=(10,10),
    colormap='plasma',
    save_format='png',
    dpi = 300,
    data_point_size=30,
    filename = '2D',
    xy_range = None
):
    """
    Generates a 2D scatter plot of an embedding and optionally saves it to a file.

    Parameters:
        embedding (np.ndarray): 2D array of shape (n_samples, 2) representing the projected points.
        class_label (np.ndarray, optional): Array of class labels for coloring points. If None, all points are plotted in the same color.
        background_color (str, optional): Color of the plot background. Default is 'white'.
        output_path (str, optional): File path to save the plot. If None, the plot is displayed instead of being saved.
        figsize (tuple, optional): Size of the figure (width, height). Default is the global `(10,10)`.
        colormap (str or matplotlib.colors.Colormap, optional): Colormap to use for different classes. Default is `plasma`.
        save_format (str, optional): Format to save the figure ('png', 'pdf', etc.). Default is `png`.
        dpi (int, optional): Resolution in dots per inch when saving the figure. Default is `600`.
        data_point_size (float, optional): Size of the scatter plot markers. Default is `30`.

    Notes:
        - If `class_label` is provided, each unique class will be assigned a distinct color.
        - Axes ticks are removed, but spines are kept visible.
        - The plot uses equal aspect ratio and hides the axis for a cleaner visualization.
    """
    plt.rcParams["figure.dpi"] = dpi      # display resolution
    plt.rcParams["savefig.dpi"] = dpi     # saved file resolution
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor(background_color)

    if xy_range == None:
        xy_range = x_y_range(embedding)
    
    
    
    # ----- Flexible scatter plot -----

    # Case A: User gives a SINGLE COLOR (not a colormap)
    # --------------------------------------------------
    if isinstance(colormap, str) and colormap.lower() in mcolors.CSS4_COLORS:
        single_color = colormap
        use_single_color = True

    # Case B: User gives a HEX color (e.g. "#00FF00")
    elif isinstance(colormap, str) and colormap.startswith("#") and len(colormap) in (4, 7):
        single_color = colormap
        use_single_color = True

    else:
        # Otherwise assume it is a COLORMAP
        # cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
        cmap = colormaps.get_cmap(colormap) if isinstance(colormap, str) else colormap
        use_single_color = False

    # ---- PLOT WITH CLASS COLORS ----
    if class_label is not None:

        unique_classes = np.unique(class_label)
        num_classes = len(unique_classes)
        legend_elements = []

        if use_single_color:
            # All classes → same color
            for cls in unique_classes:
                mask = class_label == cls
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    color=single_color,
                    # edgecolor="k",
                    edgecolor=colors[ind],
                    s=data_point_size,
                    zorder=5,
                    facecolors = 'none',
                    linewidths = 0.75,
                    # linestyle = '--',
                )
        else:
            # Use colormap for classes
            colors = [cmap(i / max(1, num_classes - 1)) for i in range(num_classes)]
            for ind, cls in enumerate(unique_classes):
                mask = class_label == cls
                
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    color=colors[ind],
                    edgecolor=colors[ind],
                    s=data_point_size,
                    linewidths=0.75,
                    zorder=5,
                    facecolors='none'
                )

                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                    label=str(cls),
                    markerfacecolor=colors[ind],
                    markeredgecolor='k',
                    markersize=8)
                )
            ax.legend(
                handles=legend_elements,
                title="Classes",
                loc="best",
                frameon=True
            )

    else:
        # ---- NO CLASS LABEL ----
        if use_single_color:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                color=single_color,
                edgecolor="k",
                s=data_point_size,
                zorder=5,
            )
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                color=cmap(0.5),
                edgecolor="k",
                s=data_point_size,
                zorder=5,
            )


    # Remove ticks, keep spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)
    ax.set_aspect("auto")
    ax.axis("off")
  

    # Save or show
    if output_path:
        output_file = os.path.join(output_path, f"projection_low_dm_{filename}.{save_format}")
        plt.savefig(output_file, bbox_inches="tight")  # used in paper
        plt.close()
    else:
        plt.show()



def plot_projection_custom_color(
    embedding,
    class_label=None,
    color_list=None,   # ONLY HEX LIST
    background_color="white",
    output_path=None,
    figsize=(10,10),
    save_format='png',
    dpi=300,
    data_point_size=30,
    linewidths = 0.2,
    legend_dt_point_size = 500,
    legend_font_size = 15,
    filename='2D',
    xy_range=None,
    class_names = None,
):

    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.distiller.res'] = 6000  # high-quality raster parts


    # ---------------- DPI SETTINGS ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor(background_color)

    # ---------------- XY RANGE ----------------
    if xy_range is None:
        xy_range = x_y_range(embedding)

    # ---------------- DEFAULT COLOR LIST ----------------
    if color_list is None:
        color_list = [
            "#00FFFF", "#A855F7", "#00FF00", "#D619B6",
            "#00A6FB", "#E33EDE", "#C026D3", "#DB2777",
            "#018700", "#56642A", "#FF6F00", "#00BFA5",
            "#8E24AA", "#3949AB", "#039BE5", "#43A047",
            "#FDD835", "#FB8C00", "#6D4C41", "#546E7A"
        ]

    # ==================================================
    # ---------------- SCATTER PLOT ---------------------
    # ==================================================
   
    if class_label is not None:

        # -------- Separate noise FIRST (CRITICAL FIX) --------
        noise_mask = class_label == -1
        valid_mask = class_label != -1

        valid_labels = class_label[valid_mask]
        unique_classes = np.unique(valid_labels)
        unique_classes = np.sort(unique_classes)

        n_colors = len(color_list)

        # -------- Stable mapping (ONLY valid classes) --------
        cluster_color_map = {
            cls: color_list[i % n_colors]
            for i, cls in enumerate(unique_classes)
        }

        legend_elements = []

        # -------- Plot ONLY valid clusters --------
        for cls in unique_classes:
            mask = class_label == cls
            color = cluster_color_map[cls]

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=color,
                edgecolor=color,
                s=data_point_size,
                linewidths=linewidths,
                zorder=5,
                facecolors='none'
            )

            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=str(cls),
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=8
                )
            )

        # -------- Plot noise ONLY once --------
        if np.any(noise_mask):
            noise_color = "#453D3DF7"

            ax.scatter(
                embedding[noise_mask, 0],
                embedding[noise_mask, 1],
                color=noise_color,
                edgecolor=noise_color,
                s=data_point_size,
                linewidths=0.5,
                alpha=0.8,
                zorder=3,
                facecolors='none'
            )

            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label="Noise",
                    markerfacecolor=noise_color,
                    markeredgecolor=noise_color,
                    markersize=8
                )
            )

        ax.legend(
            handles=legend_elements,
            title="Classes",
            loc="best",
            frameon=True
        )
    # ---------------- NO LABEL CASE ----------------
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            color=color_list[0],
            edgecolor=color_list[0],
            s=data_point_size,
            zorder=5,
            facecolors='none'
        )

    if class_names is not None:
        for name, color in zip(class_names, color_list):
            plt.scatter([], [], color=color, label=name, s=legend_dt_point_size)

        plt.legend(loc='upper center',
                bbox_to_anchor=(0.5, 1.10),
                ncol=len(class_names),
                frameon=False,
                fontsize=legend_font_size,
                handletextpad=0.05,   # space between dot and text (↓ reduce)
                columnspacing=0.5,   # space between columns (↓ reduce)
                borderaxespad=0.2,   # padding around legend
                labelspacing=0.3     # vertical spacing (if wraps)
                )

    # ==================================================
    # ---------------- FINAL STYLING --------------------
    # ==================================================
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)

    ax.set_aspect("auto")
    ax.axis("off")

    # ==================================================
    # ---------------- SAVE / SHOW ----------------------
    # ==================================================
    if output_path:
        output_file = os.path.join(
            output_path,
            # f"projection_low_dm_{filename}.{save_format}"
            f"projection_low_dm_{filename}"
        )
        plt.savefig(output_file, bbox_inches="tight")
        # plt.savefig(f"{output_file}.eps", format="eps", dpi=12000, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()



def plot_checkviz(
    low_dm_emb,
    false_distortion_list,
    missing_distortion_list,
    class_label = None,
    output_path = None,
    filename = "checkviz_original",
    colormap_clust= 'plasma',
    figsize = (10,10),
    data_point_size = 30,
    save_format = 'png',
    dpi = 300,
    bscatter_plot = True,
    xy_range = None
):
    
    if xy_range == None:
        xy_range = x_y_range(low_dm_emb)

    vor = Voronoi(low_dm_emb)

    plt.rcParams["figure.dpi"] = dpi      # display resolution
    plt.rcParams["savefig.dpi"] = dpi     # saved file resolution
    fig, ax = plt.subplots(figsize=figsize)
       

    cell_colors = []
    for idx, region in enumerate(vor.regions[:-1]):
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            color = checkviz_cmap(
                false_distortion_list[idx], missing_distortion_list[idx]
            )
            cell_colors.append(color)
            ax.fill(*zip(*polygon), color)

    # ----- Flexible scatter plot -----
    if bscatter_plot:

        # Case A: User gives a SINGLE COLOR (not a colormap)
        # --------------------------------------------------
        if isinstance(colormap_clust, str) and colormap_clust.lower() in mcolors.CSS4_COLORS:
            single_color = colormap_clust
            use_single_color = True

        # Case B: User gives a HEX color (e.g. "#00FF00")
        elif isinstance(colormap_clust, str) and colormap_clust.startswith("#") and len(colormap_clust) in (4, 7):
            single_color = colormap_clust
            use_single_color = True

        else:
            # Otherwise assume it is a COLORMAP
            cmap = colormaps.get_cmap(colormap_clust) if isinstance(colormap_clust, str) else colormap_clust
            use_single_color = False

        # ---- PLOT WITH CLASS COLORS ----
        if class_label is not None:

            unique_classes = np.unique(class_label)
            num_classes = len(unique_classes)

            if use_single_color:
                # All classes → same color
                for cls in unique_classes:
                    mask = class_label == cls
                    ax.scatter(
                        low_dm_emb[mask, 0],
                        low_dm_emb[mask, 1],
                        color=single_color,
                        edgecolor="k",
                        s=data_point_size,
                        zorder=5,
                    )
            else:
                # Use colormap for classes
                colors = [cmap(i / max(1, num_classes - 1)) for i in range(num_classes)]
                for ind, cls in enumerate(unique_classes):
                    mask = class_label == cls
                    
                    ax.scatter(
                        low_dm_emb[mask, 0],
                        low_dm_emb[mask, 1],
                        color=colors[ind],
                        edgecolor=colors[ind],
                        s=data_point_size,
                        linewidths=0.75,
                        zorder=5,
                        facecolors='none'
                    )

        else:
            # ---- NO CLASS LABEL ----
            if use_single_color:
                ax.scatter(
                    low_dm_emb[:, 0],
                    low_dm_emb[:, 1],
                    color=single_color,
                    edgecolor="k",
                    s=data_point_size,
                    zorder=5,
                )
            else:
                ax.scatter(
                    low_dm_emb[:, 0],
                    low_dm_emb[:, 1],
                    color=cmap(0.5),
                    edgecolor="k",
                    s=data_point_size,
                    zorder=5,
                )


    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # plt.tight_layout()

    # Save or display
    if output_path:
        output_path = os.path.join(
                            output_path, filename
                                    )
        plt.savefig(f"{output_path}.{save_format}", bbox_inches="tight" , pad_inches=0)

        plt.close()
    else:
        plt.show()
        plt.close()


def plot_checkviz_custom_color(
    low_dm_emb,
    false_distortion_list,
    missing_distortion_list,
    class_label=None,
    color_list=None,   # ONLY color list
    output_path=None,
    filename="checkviz_original",
    figsize=(10, 10),
    data_point_size=30,
    save_format="png",
    dpi=300,
    bscatter_plot=True,
    xy_range=None
):
    """
    Clean CheckViz plot using ONLY color list for clusters.

    Parameters
    ----------
    color_list : list of HEX colors (REQUIRED if class_label is given)
    """

    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.distiller.res'] = 6000  # high-quality raster parts

    # ---------------- DPI SETTINGS ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    # ---------------- RANGE ----------------
    if xy_range is None:
        xy_range = x_y_range(low_dm_emb)

    # ---------------- FIGURE ----------------
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    fig, ax = plt.subplots(figsize=figsize)

    # ---------------- VORONOI ----------------
    vor = Voronoi(low_dm_emb)

    for idx, region in enumerate(vor.regions[:-1]):
        if not region or -1 in region:
            continue

        polygon = [vor.vertices[i] for i in region]

        color = checkviz_cmap(
            false_distortion_list[idx],
            missing_distortion_list[idx]
        )

        ax.fill(*zip(*polygon), color=color)

    # ---------------- SCATTER ----------------
    if bscatter_plot:

        if class_label is not None:

            if color_list is None:
                raise ValueError("You must provide a color_list when using class_label.")

            unique_classes = np.unique(class_label)
            unique_classes = unique_classes[unique_classes != -1]  # remove noise

            # FIXED COLOR MAPPING (ORDER PRESERVED)
            cluster_color_map = {
                cls: color_list[i % len(color_list)]
                for i, cls in enumerate(unique_classes)
            }

            # ---- Plot clusters ----
            for cls in unique_classes:
                mask = class_label == cls

                ax.scatter(
                    low_dm_emb[mask, 0],
                    low_dm_emb[mask, 1],
                    facecolors='none',
                    edgecolors=cluster_color_map[cls],
                    s=data_point_size,
                    linewidths=0.75,
                    zorder=5
                )

            # ---- Plot noise separately ----
            noise_mask = class_label == -1
            if np.any(noise_mask):
                ax.scatter(
                    low_dm_emb[noise_mask, 0],
                    low_dm_emb[noise_mask, 1],
                    facecolors='none',
                    edgecolors='gray',
                    s=data_point_size,
                    linewidths=0.75,
                    zorder=3
                )

        else:
            # No class labels → single neutral color
            ax.scatter(
                low_dm_emb[:, 0],
                low_dm_emb[:, 1],
                facecolors='none',
                edgecolors='black',
                s=data_point_size,
                linewidths=0.75,
                zorder=5
            )

    # ---------------- FINAL STYLING ----------------
    ax.set_xlim(xy_range.x_min, xy_range.x_max)
    ax.set_ylim(xy_range.y_min, xy_range.y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- SAVE / SHOW ----------------
    if output_path:
        full_path = os.path.join(output_path, filename)
        plt.savefig(f"{full_path}.{save_format}", bbox_inches="tight", pad_inches=0)
        # plt.savefig(f"{full_path}.eps", format="eps", dpi=12000, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


###__________________3D Cubic_________________________________________________________________

def plotly_3d_cubes_colormap(data, labels, filename="cubic", output_path=None, colormap='plasma'):
    unique_labels = np.unique(labels)
    # cmap = cm.get_cmap(colormap, len(unique_labels))
    cmap = colormaps.get_cmap(colormap).resampled(len(unique_labels))
    fig = go.Figure()

    for i, lbl in enumerate(unique_labels):
        pts = data[labels == lbl]
        color = f"rgba{tuple(int(255*c) for c in cmap(i)[:3]) + (0.8,)}"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=f'Cube {i+1}'
        ))

    fig.update_layout(
        title="Interactive 3D Multi-Cube Dataset (Distinct Plasma Colors)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800, height=700,
        showlegend=True
    )
    if output_path:
        output_file = os.path.join(output_path, f"3D_{filename}.html")
        fig.write_html(output_file)
        print(f"Interactive 3D plot saved to: {output_file}")

    fig.show()


def plotly_3d_cubes_custom_colormap(
    data,
    labels,
    filename="cubic",
    output_path=None,
    custom_colors=None,
    alpha=1.0
):

    unique_labels = np.unique(labels)
    fig = go.Figure()

    # Default colors
    if custom_colors is None:
        custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Convert colors to RGB (0–255)
    colors = []
    for c in custom_colors:
        if isinstance(c, str):
            r, g, b = to_rgb(c)
            colors.append((int(r*255), int(g*255), int(b*255)))
        elif isinstance(c, (tuple, list)) and len(c) == 3:
            colors.append(c)
        else:
            raise ValueError(f"Invalid color format: {c}")

    # Plot points
    for i, lbl in enumerate(unique_labels):
        pts = data[labels == lbl]

        r, g, b = colors[i % len(colors)]
        color = f"rgba({r},{g},{b},{alpha})"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='rgba(0,0,0,0)',  # transparent fill
                line=dict(
                    color=color,
                    width=1.5
                )
            ),
            name=f'Cluster {i+1}'
        ))

    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
                showticklabels=False,
                ticks='',
                showgrid=False,
                zeroline=True,
                showline=True,
                showbackground=True,
                linecolor='black'
            ),
            yaxis=dict(
                title="Y",
                showticklabels=False,
                ticks='',
                showgrid=False,
                zeroline=True,
                showline=True,
                showbackground=True,
                linecolor='black'
            ),
            zaxis=dict(
                title="Z",
                showticklabels=False,
                ticks='',
                showgrid=False,
                zeroline=True,
                showline=True,
                showbackground=True,
                linecolor='black'
            )
        )
    )

    # Save outputs
    if output_path:
        output_file = os.path.join(output_path, f"3D_{filename}.html")
        fig.write_html(output_file)
        print(f"Saved to: {output_file}")
    fig.show()


#_____________________________

#________________ Sensitivity & Robustness Analysis-----------------------------------------------------
def plot_rank_comparison_fast(
    X_R2,
    res,
    top_k=20,
    background_percentile=80,
    max_background_lines=3000,
    max_scatter_points=50000,
    show_labels=False,
    save_path=None,
):

    pairs = res["pairs"]
    rank_Rn = res["rank_Rn"]
    rank_R2 = res["rank_R2"]
    shift = res["shift"]
    abs_shift = res["abs_shift"]
    n_pairs = res["n_pairs"]
    rho = res["rho"]

    # =====================================================
    # Colormaps
    # =====================================================

    norm_abs = mcolors.Normalize(
        vmin=0,
        vmax=abs_shift.max()
    )

    cmap_abs = plt.cm.plasma

    norm_div = mcolors.TwoSlopeNorm(
        vcenter=0,
        vmin=shift.min(),
        vmax=shift.max()
    )

    cmap_div = plt.cm.RdBu_r

    # =====================================================
    # Figure
    # =====================================================

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#0f1117")

    axes_color = "#1a1d27"
    text_color = "#e8e8f0"
    grid_color = "#2a2d3a"

    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.38,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    ax = [
        fig.add_subplot(gs[r, c])
        for r in range(2)
        for c in range(2)
    ]

    for a in ax:

        a.set_facecolor(axes_color)

        for spine in a.spines.values():
            spine.set_edgecolor(grid_color)

        a.tick_params(
            colors=text_color,
            labelsize=9
        )

        a.xaxis.label.set_color(text_color)
        a.yaxis.label.set_color(text_color)
        a.title.set_color(text_color)

    # =====================================================
    # A. Rank-Rank Scatter
    # =====================================================

    if n_pairs > max_scatter_points:

        sample_idx = np.random.choice(
            n_pairs,
            max_scatter_points,
            replace=False
        )

    else:

        sample_idx = np.arange(n_pairs)

    sc = ax[0].scatter(
        rank_Rn[sample_idx],
        rank_R2[sample_idx],
        c=abs_shift[sample_idx],
        cmap=cmap_abs,
        norm=norm_abs,
        s=6,
        alpha=0.7,
        linewidths=0,
        rasterized=True
    )

    diag = [1, n_pairs]

    ax[0].plot(
        diag,
        diag,
        color="#5588ff",
        lw=1.2,
        ls="--",
        alpha=0.6,
        label="perfect rank preservation"
    )

    ax[0].set_xlabel("Rank in Rⁿ")
    ax[0].set_ylabel("Rank in R²")

    ax[0].set_title(
        f"A — Rank-rank scatter (ρ={rho:.3f})",
        fontweight="bold"
    )

    ax[0].legend(
        fontsize=8,
        facecolor=axes_color,
        labelcolor=text_color,
        framealpha=0.8
    )

    cb = fig.colorbar(
        sc,
        ax=ax[0],
        fraction=0.046,
        pad=0.04
    )

    cb.set_label(
        "|rank shift|",
        color=text_color,
        fontsize=8
    )

    ax[0].grid(
        color=grid_color,
        lw=0.5
    )

    # =====================================================
    # B. Histogram
    # =====================================================

    bins = min(
        60,
        n_pairs // 20 + 10
    )

    n_hist, bin_edges, patches = ax[1].hist(
        shift,
        bins=bins,
        edgecolor="none"
    )

    centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2

    for patch, center in zip(
        patches,
        centers
    ):
        patch.set_facecolor(
            cmap_div(norm_div(center))
        )
        patch.set_alpha(0.85)

    ax[1].axvline(
        0,
        color="#5588ff",
        lw=1.5,
        ls="--"
    )

    mean_abs = abs_shift.mean()

    ax[1].set_title(
        f"B — Rank shift distribution "
        f"(mean |shift|={mean_abs:.1f})",
        fontweight="bold"
    )

    ax[1].set_xlabel(
        "Rank shift (R² − Rⁿ)"
    )

    ax[1].set_ylabel(
        "Number of pairs"
    )

    ax[1].grid(
        axis="y",
        color=grid_color,
        lw=0.5
    )

    # =====================================================
    # C. Projection
    # =====================================================

    threshold = np.percentile(
        abs_shift,
        background_percentile
    )

    background_idx = np.where(
        abs_shift < threshold
    )[0]

    if len(background_idx) > max_background_lines:

        background_idx = np.random.choice(
            background_idx,
            max_background_lines,
            replace=False
        )

    # ---- background lines ----

    segments = []

    for idx in background_idx:

        i, j = pairs[idx]

        segments.append([
            X_R2[i],
            X_R2[j]
        ])

    lc = LineCollection(
        segments,
        colors="#333655",
        linewidths=0.3,
        alpha=0.25
    )

    ax[2].add_collection(lc)

    # ---- top-k disrupted ----

    worst_idx = np.argsort(
        abs_shift
    )[::-1][:top_k]

    top_segments = []
    top_colors = []

    for idx in worst_idx:

        i, j = pairs[idx]

        top_segments.append([
            X_R2[i],
            X_R2[j]
        ])

        top_colors.append(
            cmap_div(
                norm_div(
                    shift[idx]
                )
            )
        )

    lc_top = LineCollection(
        top_segments,
        colors=top_colors,
        linewidths=2.0,
        alpha=0.9
    )

    ax[2].add_collection(lc_top)

    # ---- points ----

    ax[2].scatter(
        X_R2[:, 0],
        X_R2[:, 1],
        color="#f0e0ff",
        s=25,
        edgecolors="#8888bb",
        lw=0.4,
        zorder=5
    )

    # ---- labels (optional) ----

    if show_labels:

        for idx in range(len(X_R2)):

            ax[2].text(
                X_R2[idx, 0],
                X_R2[idx, 1],
                str(idx),
                fontsize=6,
                color="#ccccee",
                ha="center",
                va="center",
                zorder=6
            )

    ax[2].autoscale()

    ax[2].set_title(
        f"C — Top {top_k} most disrupted pairs",
        fontweight="bold"
    )

    ax[2].set_xlabel("R² dim 1")
    ax[2].set_ylabel("R² dim 2")

    ax[2].grid(
        color=grid_color,
        lw=0.4,
        alpha=0.5
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap_div,
        norm=norm_div
    )

    sm.set_array([])

    cb2 = fig.colorbar(
        sm,
        ax=ax[2],
        fraction=0.046,
        pad=0.04
    )

    cb2.set_label(
        "signed rank shift",
        color=text_color,
        fontsize=8
    )

    # =====================================================
    # D. CDF
    # =====================================================

    sorted_abs = np.sort(abs_shift)

    cdf = (
        np.arange(
            1,
            n_pairs + 1
        )
        / n_pairs
    )

    ax[3].plot(
        sorted_abs,
        cdf,
        color="#a0d0ff",
        lw=2
    )

    ax[3].fill_between(
        sorted_abs,
        cdf,
        alpha=0.15
    )

    for pct, ls in [
        (50, "--"),
        (90, ":")
    ]:

        val = np.percentile(
            abs_shift,
            pct
        )

        ax[3].axvline(
            val,
            color="#ffb347",
            lw=1.2,
            ls=ls,
            label=f"p{pct}={val:.0f}"
        )

    ax[3].set_title(
        "D — Cumulative distribution",
        fontweight="bold"
    )

    ax[3].set_xlabel("|Rank shift|")
    ax[3].set_ylabel(
        "Cumulative fraction"
    )

    ax[3].set_ylim(0, 1)

    ax[3].legend(
        fontsize=8,
        facecolor=axes_color,
        labelcolor=text_color
    )

    ax[3].grid(
        color=grid_color,
        lw=0.5
    )

    # =====================================================
    # Save
    # =====================================================

    fig.suptitle(
        "Rank-distance preservation: Rⁿ → R²",
        fontsize=15,
        fontweight="bold",
        color=text_color
    )

    if save_path:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

        print(f"Figure saved → {save_path}")

    plt.show()


#--------------------- RANK SHIFT---------------------------------------------------------
def plot_rank_shift_with_delaunay_fast(
    results,
    delaunay_edges,
    filename="rank_shift",
    output_path=None,
    save_format="png"
):

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    rank_Rn = np.asarray(results["rank_Rn"])
    shift   = np.asarray(results["shift"])
    pairs   = np.asarray(results["pairs"])

    # --------------------------------------------------
    # Normalize edges (i < j form)
    # --------------------------------------------------
    delaunay_edges = np.sort(np.asarray(delaunay_edges, dtype=np.int64), axis=1)
    pairs = np.sort(pairs, axis=1)

    # --------------------------------------------------
    # FAST EDGE MATCHING
    # --------------------------------------------------
    SCALE = int(1e6)

    delaunay_keys = delaunay_edges[:, 0] * SCALE + delaunay_edges[:, 1]
    pair_keys     = pairs[:, 0] * SCALE + pairs[:, 1]

    delaunay_set = set(delaunay_keys)

    edge_mask = np.fromiter(
        (k in delaunay_set for k in pair_keys),
        dtype=bool,
        count=len(pair_keys)
    )

    # --------------------------------------------------
    # Sort by rank
    # --------------------------------------------------
    idx = np.argsort(rank_Rn)

    x = rank_Rn[idx]
    y = shift[idx]
    edge_mask = edge_mask[idx]

    # --------------------------------------------------
    # Split masks
    # --------------------------------------------------
    pos_mask = (~edge_mask) & (y > 0)
    neg_mask = (~edge_mask) & (y < 0)
    del_mask = edge_mask

    # --------------------------------------------------
    # Vectorized segment builder
    # --------------------------------------------------
    def build_segments(xvals, yvals):
        return np.stack(
            [
                np.column_stack([xvals, np.zeros_like(yvals)]),
                np.column_stack([xvals, yvals])
            ],
            axis=1
        )

    seg_pos = build_segments(x[pos_mask], y[pos_mask])
    seg_neg = build_segments(x[neg_mask], y[neg_mask])
    seg_del = build_segments(x[del_mask], y[del_mask])

    # # --------------------------------------------------
    # # Optional downsampling for extremely large plots
    # # --------------------------------------------------
    # MAX_SEGMENTS = 100000

    # if len(seg_pos) > MAX_SEGMENTS:
    #     step = max(1, len(seg_pos) // MAX_SEGMENTS)
    #     seg_pos = seg_pos[::step]

    # if len(seg_neg) > MAX_SEGMENTS:
    #     step = max(1, len(seg_neg) // MAX_SEGMENTS)
    #     seg_neg = seg_neg[::step]

    # if len(seg_del) > MAX_SEGMENTS:
    #     step = max(1, len(seg_del) // MAX_SEGMENTS)
    #     seg_del = seg_del[::step]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)

    # ---------------- Positive shifts ----------------
    if len(seg_pos):
        lc_pos = LineCollection(
            seg_pos,
            colors="indianred",
            linewidths=0.4,
            alpha=0.6,
            zorder=1,
            rasterized=True
        )
        # lc_pos.set_rasterized(True)
        ax.add_collection(lc_pos)

    # ---------------- Negative shifts ----------------
    if len(seg_neg):
        lc_neg = LineCollection(
            seg_neg,
            colors="royalblue",
            linewidths=0.4,
            alpha=0.6,
            zorder=1,
            rasterized=True
        )
        # lc_neg.set_rasterized(True)
        ax.add_collection(lc_neg)

    # ---------------- Delaunay edges ----------------
    if len(seg_del):
        lc_del = LineCollection(
            seg_del,
            colors="white",
            linewidths=0.4,
            alpha=1.0,
            zorder=10,
            rasterized=True
        )
        # lc_del.set_rasterized(True)
        ax.add_collection(lc_del)

    # --------------------------------------------------
    # Decorations
    # --------------------------------------------------
    ax.axhline(
        0,
        color="white",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(r"Rank in $R^n$", fontsize=14)
    ax.set_ylabel("Rank shift", fontsize=14)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    legend_elements = [
        Line2D(
            [0], [0],
            color='indianred',
            lw=1.5,
            label='Positive rank shift'
        ),
        Line2D(
            [0], [0],
            color='royalblue',
            lw=1.5,
            label='Negative rank shift'
        ),
        Line2D(
            [0], [0],
            color='white',
            lw=1.5,
            label='Delaunay edges'
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fontsize=12
    )

    # ax.autoscale()
    # plt.tight_layout()

    # --------------------------------------------------
    # Save / Show
    # --------------------------------------------------
    if output_path:

        os.makedirs(output_path, exist_ok=True)

        base = os.path.join(output_path, filename)

        # User requested format
        fig.savefig(
            f"{base}.{save_format}",
            dpi=300,
            bbox_inches="tight"
        )

        # # Compact PDF (recommended)
        # plt.savefig(
        #     f"{base}.pdf",
        #     bbox_inches="tight"
        # )

        # # Compact EPS
        # fig.savefig(
        #     f"{base}.eps",
        #     format="eps",
        #     dpi=150,
        #     bbox_inches="tight",
        #     # bbox_inches=None,
        #     pad_inches=0.1,
        #     # backend="ps"
        # )
        plt.show()
        plt.close()

    else:
        plt.show()
        plt.close()


