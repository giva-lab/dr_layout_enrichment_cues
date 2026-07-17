import matplotlib.tri as tri
import numpy as np
from scipy.spatial import Delaunay


from .utility import (barycentric_interpolation,
                      calculate_delanay_edge_length,
                      calculate_NLM_CCA_probability, checkviz_cmap,
                      hex_to_rgb_normalized_or_nan, 
                      x_y_range, get_reducer, 
                      inter_intra_cluster_pairwise_distance_optimised)


class ProjectionAnalysisVisualizer:
    def __init__(
        self, data, embedding = None, class_label=None, projection_method = None, perplexity=30, num_grid_points=500, output_path=None
    ):
        """
        Initialize the ProjectionAnalysisVisualizer.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)
            High-dimensional dataset (original data).
        embedding : np.ndarray of shape (n_samples, 2), optional
            2D embedding coordinates of the data (projected data). 
            If None, a projection is computed using `projection_method`.
        class_label : np.ndarray of shape (n_samples,) or None, optional
            Class labels for each point. Used for supervised visualizations.
        projection_method : str or None, optional
            Dimensionality reduction method (e.g., 'tsne', 'umap').
            Required if `embedding` is None.
        num_grid_points : int, optional
            Number of grid points for interpolation and analysis in 2D. Default is 500.
        output_path : str or None, optional
            Directory path for saving plots and results. Default is None.

        Raises
        ------
        ValueError
            If neither `embedding` nor `projection_method` is provided.
        """

        self.D = data
        self.reducer = get_reducer(method=projection_method, perplexity=perplexity) if projection_method else None
        self.embedding = embedding if embedding is not None else (self.reducer.fit_transform(self.D) if self.reducer else (_ for _ in ()).throw(ValueError("Either 'embedding' or 'projection_method' must be provided.")))
        self.class_label = class_label
        self.num_grid_points = num_grid_points
        self.inverse_model = None
        self.tri_delaunay = Delaunay(self.embedding)
        self.tri_nodes = self.tri_delaunay.simplices

        # Create a triangulation object using the t-SNE coordinates
        self.triang_t_sne = tri.Triangulation(
            self.embedding[:, 0], self.embedding[:, 1], self.tri_nodes
        )
        self.all_tri_edges_len_hd, self.all_tri_edges_len_ld = (
            calculate_delanay_edge_length(self.D, self.embedding, self.tri_nodes)
        )

        self.xy_range = x_y_range(self.embedding, margin=0.05)
        self.x_min, self.x_max = np.min(self.embedding[:, 0]), np.max(self.embedding[:, 0])
        self.y_min, self.y_max = np.min(self.embedding[:, 1]), np.max(self.embedding[:, 1])
        self.P_CCA_norm = None
        self.P_NLM_norm = None
        self.P_CCA_interp = None
        self.P_NLM_interp = None
        self.output_path = output_path  # folder path where to save

    def compute_distortion_prob(self, data = None, embedding = None, sigma=None):
        """
        Compute Normalized Local Metric (NLM) and Class Consistency Agreement (CCA) 
        probabilities for each data point.

        Parameters
        ----------
        data : np.ndarray, optional
            High-dimensional input data. Defaults to `self.D`.
        embedding : np.ndarray, optional
            2D embedding coordinates. Defaults to `self.embedding`.
        sigma : float or None, optional
            Gaussian kernel width for local neighborhood computation. 
            If None, the distance to the 5th nearest neighbor is used.

        Returns
        -------
        P_NLM_norm : np.ndarray
            Normalized NLM probabilities per data point.
        P_CCA_norm : np.ndarray
            Normalized CCA probabilities per data point.
        """

        data = data if data is not None else self.D
        embedding = embedding if embedding is not None else self.embedding

        self.P_NLM_norm, self.P_CCA_norm = calculate_NLM_CCA_probability(
            data, embedding, sigma=sigma
        )

        return self.P_NLM_norm, self.P_CCA_norm

    def interpolate_distortion_metrics(self, P_CCA_norm = None, P_NLM_norm = None ):
        """
        Interpolate normalized CCA and NLM probabilities over the 2D embedding grid 
        using barycentric interpolation over Delaunay Triangulation.

        Parameters
        ----------
        P_CCA_norm : np.ndarray, optional
            Precomputed normalized CCA probabilities. Defaults to `self.P_CCA_norm`.
        P_NLM_norm : np.ndarray, optional
            Precomputed normalized NLM probabilities. Defaults to `self.P_NLM_norm`.

        Returns
        -------
        P_NLM_interp : np.ndarray of shape (H, W)
            Interpolated NLM probabilities on the grid.
        P_CCA_interp : np.ndarray of shape (H, W)
            Interpolated CCA probabilities on the grid.
        """

        P_CCA_norm = P_CCA_norm if P_CCA_norm is not None else self.P_CCA_norm
        P_NLM_norm = P_NLM_norm if P_NLM_norm is not None else self.P_NLM_norm

        self.P_CCA_interp = barycentric_interpolation(
            self.num_grid_points,
            self.embedding,
            self.tri_delaunay,
            P_CCA_norm,
            self.xy_range,
            bVertex_based=True,
            blog=False,
            bclamping=False,
        )

        self.P_NLM_interp = barycentric_interpolation(
            self.num_grid_points,
            self.embedding,
            self.tri_delaunay,
            P_NLM_norm,
            self.xy_range,
            bVertex_based=True,
            blog=False,
            bclamping=False,
        )

        return self.P_NLM_interp, self.P_CCA_interp

    def safe_checkviz_map(self, pcca, pnml):
        """
        Map CCA and NLM probabilities to a color safely.

        Parameters
        ----------
        pcca : float
            Single CCA probability value.
        pnml : float
            Single NLM probability value.

        Returns
        -------
        color_code : str or np.nan
            Hexadecimal color code from `checkviz_cmap`, or NaN if input is invalid.
        """
        if np.isnan(pcca) or np.isnan(pnml):
            return np.nan
        return checkviz_cmap(pcca, pnml)

    def compute_grid_colors(self, P_NLM_interp = None, P_CCA_interp = None):
        """
        Convert interpolated probability maps to RGB colors using the CheckViz colormap.

        Parameters
        ----------
        P_NLM_interp : np.ndarray, optional
            Interpolated NLM probabilities. Defaults to `self.P_NLM_interp`.
        P_CCA_interp : np.ndarray, optional
            Interpolated CCA probabilities. Defaults to `self.P_CCA_interp`.

        Returns
        -------
        rgb_array : np.ndarray of shape (H, W, 3)
            RGB colors corresponding to the grid points.
        """
        
        P_NLM_interp = P_NLM_interp if P_NLM_interp is not None else self.P_NLM_interp
        P_CCA_interp = P_CCA_interp if P_CCA_interp is not None else self.P_CCA_interp

        H, W = P_NLM_interp.shape
        grid_colors = np.empty((H, W), dtype=object)

        for i in range(H):
            for j in range(W):
                grid_colors[i, j] = self.safe_checkviz_map(
                    P_CCA_interp[i, j], P_NLM_interp[i, j]
                )

        flat_colors = grid_colors.ravel()
        rgb_flat = np.array(
            [hex_to_rgb_normalized_or_nan(c) for c in flat_colors], dtype=np.float32
        )
        rgb_array = rgb_flat.reshape(H, W, 3)

        return rgb_array

    
    def delaunay_triangulation(self, embedding = None):

        """
        Compute and return the Delaunay triangulation of the embedding.

        Parameters
        ----------
        embedding : ndarray, optional
            2D array of points with shape (n_samples, 2). 
            If None, uses `self.embedding`.

        Returns
        -------
        tri_delaunay : scipy.spatial.Delaunay
            Delaunay triangulation object containing information about simplices,
            neighbors, and convex hull of the input points.

        Notes
        -----
        - Stores the computed triangulation in `self.tri_delaunay`.  
        - If `embedding` is not provided, the method defaults to using 
        the object's internal `self.embedding`.  
        """

        embedding = embedding if embedding is not None else self.embedding

        self.tri_delaunay =Delaunay(embedding)
        return self.tri_delaunay
    
    def projection_emb_low_dim(self):
        """
        Return the low-dimensional embedding.

        Returns
        -------
        embedding : np.ndarray of shape (n_samples, 2)
            2D embedding coordinates.
        """
        
        return self.embedding
    
    def inter_intra_cluster_pairwise_distance(self, data, class_label, metric = 'euclidean', norm_distance = True, norm_type='global'):
        """
        Compute intra- and inter-cluster pairwise distances.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional dataset.
        class_label : np.ndarray
            Class labels for each data point.
        metric : str, optional
            Distance metric (default: 'euclidean').
        norm_distance : bool, optional
            Whether to normalize distances. Default is True.
        norm_type : str, optional
            Normalization type, e.g., 'global'. Default is 'global'.

        Returns
        -------
        pairwise_dist : np.ndarray
            Pairwise distances.
        mean_pairwise_dist : np.ndarray
            Mean pairwise distances per cluster.
        """

        data = data if data is not None else self.D
        class_label = class_label if class_label is not None else self.class_label
        
        pairwise_dist, mean_pairwise_dist = inter_intra_cluster_pairwise_distance_optimised(data, class_label, metric = metric, norm_distance = norm_distance)

        return pairwise_dist, mean_pairwise_dist
    

    
    def delanay_hd_edge_lenghts_inter(self, output_path=None):
        """
        Interpolate high-dimensional edge lengths from the Delaunay triangulation.

        Parameters
        ----------
        output_path : str or None, optional
            Path to save the plot. Default is None.

        Returns
        -------
        intensity_interp_hd_lengths : np.ndarray
            Interpolated edge length intensities.
        """

        output_path = output_path if output_path is not None else self.output_path

        intensity_interp_hd_lengths = barycentric_interpolation(
            self.num_grid_points,
            self.embedding,
            self.tri_delaunay,
            self.all_tri_edges_len_hd,
            self.xy_range,
            blog=False,
            bclamping=False,
        )

        return intensity_interp_hd_lengths
