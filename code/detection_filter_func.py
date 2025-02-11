import numpy as np
from skimage.measure import regionprops


def filter_mcs_candidates(
    clusters, convective_plumes, min_area_km2, min_nr_plumes, grid_cell_area_km2
):
    """
    Filter clusters to identify MCS candidates based on area and number of convective plumes.

    Parameters:
    - clusters: 2D array of cluster labels.
    - convective_plumes: 2D array of convective plume labels.
    - min_area_km2: Minimum area threshold for MCS candidate (in km²).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - grid_cell_area_km2: Area of a single grid cell (in km²).

    Returns:
    - mcs_candidate_labels: List of cluster labels that meet the MCS criteria.
    """
    mcs_candidate_labels = []
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        area_km2 = np.sum(cluster_mask) * grid_cell_area_km2
        plumes_in_cluster = np.unique(convective_plumes[cluster_mask])
        num_plumes = len(plumes_in_cluster[plumes_in_cluster != 0])

        if area_km2 >= min_area_km2 and num_plumes >= min_nr_plumes:
            mcs_candidate_labels.append(label_value)

    return mcs_candidate_labels


def extract_shape_features(clusters, lat, lon, grid_spacing_km):
    """Extracts shape features (e.g., area, perimeter, axes) from labeled clusters.

    Args:
        clusters (numpy.ndarray):
            2D array of integer cluster labels (0 indicates background/no cluster).
        lat (numpy.ndarray):
            2D array of latitudes, same shape as `clusters`.
        lon (numpy.ndarray):
            2D array of longitudes, same shape as `clusters`.
        grid_spacing_km (float):
            Approximate grid spacing in kilometers for converting pixel-based measurements
            (like area in pixel count) into km².

    Returns:
        dict:
            A dictionary `shape_features` mapping each nonzero cluster label to a
            dictionary of shape properties. For example:

            shape_features[label_value] = {
                "area_km2": ...,
                "perimeter_km": ...,
                "major_axis_length_km": ...,
                "minor_axis_length_km": ...,
                "aspect_ratio": ...,
                "orientation_deg": ...,
                "solidity": ...,
                "eccentricity": ...,
                "extent": ...,
                "convex_area_km2": ...,
                "circularity": ...,
            }
    """
    shape_features = {}

    labeled_clusters = clusters.astype(int)
    cluster_labels = np.unique(labeled_clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background label = 0

    for label_value in cluster_labels:
        cluster_mask = labeled_clusters == label_value
        binary_image = cluster_mask.astype(int)

        # Compute region properties via skimage
        props_list = regionprops(binary_image)
        if len(props_list) == 0:
            continue
        props = props_list[0]  # Should be only one region per label_value

        # Convert region measurements to physical units
        area = props.area * (grid_spacing_km**2)  # km²
        perimeter = props.perimeter * grid_spacing_km  # km
        major_axis_length = props.major_axis_length * grid_spacing_km
        minor_axis_length = props.minor_axis_length * grid_spacing_km

        # Derived shape features
        aspect_ratio = (
            major_axis_length / minor_axis_length if minor_axis_length != 0 else np.nan
        )
        orientation_deg = np.degrees(props.orientation) % 360
        solidity = props.solidity
        eccentricity = props.eccentricity
        extent = props.extent
        convex_area = props.convex_area * (grid_spacing_km**2)
        if perimeter != 0:
            circularity = (4.0 * np.pi * area) / (perimeter**2)
        else:
            circularity = np.nan

        # Store base shape metrics
        shape_features[label_value] = {
            "area_km2": area,
            "perimeter_km": perimeter,
            "major_axis_length_km": major_axis_length,
            "minor_axis_length_km": minor_axis_length,
            "aspect_ratio": aspect_ratio,
            "orientation_deg": orientation_deg,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "extent": extent,
            "convex_area_km2": convex_area,
            "circularity": circularity,
        }

    return shape_features


def compute_cluster_centers_of_mass(final_labeled_regions, lat, lon, precipitation):
    """Computes the precipitation-weighted center of mass for each labeled cluster.

    Args:
        final_labeled_regions (numpy.ndarray):
            2D integer array of cluster labels (excluding -1 or 0 for background).
        lat (numpy.ndarray):
            2D array of latitudes, same shape as final_labeled_regions.
        lon (numpy.ndarray):
            2D array of longitudes, same shape as final_labeled_regions.
        precipitation (numpy.ndarray):
            2D array of precipitation values, same shape as final_labeled_regions.

    Returns:
        dict:
            A dictionary mapping 'label_value' -> (center_lat, center_lon) in degrees.
            If a cluster has zero total precipitation, we store (np.nan, np.nan).
    """
    cluster_centers = {}
    unique_labels = np.unique(final_labeled_regions)
    # Exclude background label if it's 0 or -1
    unique_labels = unique_labels[unique_labels > 0]
    unique_labels = unique_labels.astype(str)  # Ensure keys are strings for JSON

    for label_value in unique_labels:
        mask = final_labeled_regions == int(label_value)
        # precipitation for this cluster only
        cluster_precip = np.where(mask, precipitation, 0.0)
        total_precip = np.sum(cluster_precip)

        if total_precip > 0:
            lat_weighted = np.sum(lat * cluster_precip)
            lon_weighted = np.sum(lon * cluster_precip)
            center_lat = lat_weighted / total_precip
            center_lon = lon_weighted / total_precip
            cluster_centers[label_value] = (center_lat, center_lon)
        else:
            # If zero total precipitation, we cannot define a weighted center
            cluster_centers[label_value] = (np.nan, np.nan)

    return cluster_centers


def classify_mcs_types(shape_features):
    """
    Classify MCS clusters into types based on shape features.

    Parameters:
    - shape_features: Dictionary of shape features per cluster.

    Returns:
    - mcs_classification: Dictionary with cluster labels as keys and MCS types as values.
    """
    mcs_classification = {}

    for label_value, features in shape_features.items():
        aspect_ratio = features["aspect_ratio"]
        area = features["area_km2"]
        circularity = features["circularity"]

        # Initialize type
        mcs_type = "Unclassified"

        # Classification rules
        if aspect_ratio >= 5 and features["major_axis_length_km"] >= 100:
            mcs_type = "Squall Line"
        elif aspect_ratio <= 2 and area >= 100000 and circularity >= 0.7:
            mcs_type = "MCC"
        elif 2 <= aspect_ratio < 5:
            mcs_type = "Linear MCS"
        else:
            mcs_type = "Other MCS"

        mcs_classification[label_value] = mcs_type

    return mcs_classification
