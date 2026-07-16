
'''
Configuration file for distortion_cues.

This module centralizes default parameters used throughout the package.
'''

# =============================================================================
# Random Seed
# =============================================================================

GLOBAL_SEED = 5

# =============================================================================
# General plotting settings
# =============================================================================

FIGSIZE = (10, 10)
DPI = 600
SAVE_FORMAT = "png"

# =============================================================================
# Visualization settings
# =============================================================================
CUSTOM_COLORS = [                    
            "#00FF00", "#E784D7","#00FFFF","#04855E",  "#00A6FB", "#7E6D7C",'#018700',"#0BECA8",
            "#9c5bda", "#663397",'#56642a', "#83e27b", "#A855F7",'#366962','#89bacf',"#f66c0f",
            "#4FF0AF", '#00d493','#85d401','#855b89',"#4FF0B0",'#366962',"#E33EDE","#ED0490",'#93ac83',
        ]

SCATTER_POINT_SIZE = 20
LINE_WIDTH = 0.8
SCATTER_PLOT = True

COLORMAP = "glasbey20"
FOLD_COLORMAP = "glasbey20"

# =============================================================================
# Analysis settings
# =============================================================================

PERTURBATION_LEVELS = [
    0.00,
    0.01,
    0.10,
    0.20,
    0.30,
]


# =============================================================================
# Default plotting settings
# =============================================================================

DEFAULT_CONFIG = {
    "border_thickness": 40,
    "fontsize": 40,
    "legend_dt_point_size": 700,
    "legend_font_size": 16,
    "data_point_size": 20,
    "linewidths": 0.8,
}

# =============================================================================
# Dataset configurations
# =============================================================================

CUBE_CONFIG = {
    "class_names": ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"],
    "colors": ["#00FF00",  "#E784D7", "#00FFFF", "#B3CB1A", "#9c5bda", "#0BECA8", "#00A6FB",  '#018700',"#663397","#7E6D7C",  "#04855E"],
    "border_thickness": 40,
    "fontsize": 40,
    "legend_dt_point_size": 700,
    "legend_font_size": 23,
    "methods": {
        "default": {
            "perplexity": 20,
            "data_point_size": 40,
            "linewidths": 1.2,
        }
    },
}

DATASET_CONFIG = {

    "har": {
        "class_names": [
            "Walk", "Up", "Down",
            "Sit", "Stand", "Lay"
        ],
        "colors": [
            "#00FF00", "#E784D7", "#00FFFF",
            "#0BECA8", "#00A6FB", "#9c5bda",
            "#018700", "#663397", "#7E6D7C",
            "#04855E"
        ],
        "border_thickness": 30,
        "fontsize": 30,
        "legend_dt_point_size": 1000,
        "legend_font_size": 20,

        "methods": {
            "tsne": {
                "perplexity": 5,
                "data_point_size": 20,
                "linewidths": 0.8,
            },
            "umap": {
                "perplexity": 5,
                "data_point_size": 10,
                "linewidths": 0.4,
            },
        },
    },

    "aedessex": {
        "class_names": ["Male", "Female"],
        "colors": [
            "#00FF00",
            "#00FFFF",
            "#DD6ECB",
            "#B3CB1A",
        ],
        "border_thickness": 30,
        "fontsize": 40,
        "legend_dt_point_size": 800,
        "legend_font_size": 30,

        "methods": {
            "tsne": {
                "perplexity": 30,
                "data_point_size": 10,
                "linewidths": 0.4,
            },
            "umap": {
                "perplexity": 5,
                "data_point_size": 10,
                "linewidths": 0.4,
            },
        },
    },

    "mnist": {
        "class_names": [str(i) for i in range(10)],
        "border_thickness": 30,
        "fontsize": 15,
        "legend_dt_point_size": 700,
        "legend_font_size": 20,

        "methods": {
            "tsne": {
                "perplexity": 20,
                "data_point_size": 20,
                "linewidths": 0.8,
            },
            "umap": {
                "perplexity": 10,
                "data_point_size": 10,
                "linewidths": 0.4,
            },
        },
    },

    "cube_same_size_same_dist": {**CUBE_CONFIG},
    "cube_diff_size_diff_dist": {**CUBE_CONFIG},
    "cube_diff_size_same_dist": {**CUBE_CONFIG},
    "cube_same_size_diff_dist": {**CUBE_CONFIG},

    "cubic": {
        "class_names": [
            "Cluster 1",
            "Cluster 2",
            "Cluster 3",
        ],
        "colors": [
            "#00FF00", "#E784D7", "#00FFFF",
            "#B3CB1A", "#9c5bda", "#0BECA8",
            "#00A6FB", "#018700", "#663397",
            "#7E6D7C", "#04855E",
        ],
        "border_thickness": 40,
        "fontsize": 40,
        "legend_dt_point_size": 700,
        "legend_font_size": 23,

        "methods": {
            "default": {
                "perplexity": 25,
                "data_point_size": 40,
                "linewidths": 1.2,
            }
        },
    },
}

def get_config(dataset, method=None):
    """
    Return the merged configuration for a dataset and optional DR method.
    """
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(DATASET_CONFIG.get(dataset, {}))

    if method is not None:
        methods = cfg.get("methods", {})
        cfg.update(methods.get(method, methods.get("default", {})))

    return cfg
