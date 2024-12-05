import numpy as np
import os
import warnings
import datetime
import xarray as xr

def handle_splitting_event(overlap_area, next_cluster_id):
    """
    Handle splitting events where one previous cluster overlaps with multiple current clusters.

    Parameters:
    - overlap_area: Dictionary mapping current cluster labels to overlap percentages.
    - next_cluster_id: Integer representing the next available cluster ID.

    Returns:
    - current_cluster_ids: Dictionary mapping current cluster labels to assigned cluster IDs.
    - next_cluster_id: Updated next_cluster_id after assigning new IDs.
    """
    current_cluster_ids = {}
    for curr_label in overlap_area.keys():
        # Assign new IDs to the split clusters
        current_cluster_ids[curr_label] = next_cluster_id
        next_cluster_id += 1
    return current_cluster_ids, next_cluster_id

def handle_merging_event(overlap_area, previous_cluster_ids):
    """
    Handle merging events where multiple previous clusters overlap with one current cluster.

    Parameters:
    - overlap_area: Dictionary mapping previous cluster IDs to overlap percentages.
    - previous_cluster_ids: Dictionary mapping previous cluster labels to cluster IDs.

    Returns:
    - assigned_id: Cluster ID assigned to the merged cluster.
    """
    # For merging, you might choose to assign a new ID or keep one of the existing IDs
    # Here, we assign a new unique ID
    assigned_id = min(overlap_area.keys())  # Or use next_cluster_id to assign a new ID
    return assigned_id

def filter_main_mcs(mcs_id_list, main_mcs_ids):
    """
    Filter tracking results to include only main MCS tracks.

    Parameters:
    - mcs_id_list: List of MCS ID arrays.
    - main_mcs_ids: List of MCS IDs identified as main MCS.

    Returns:
    - filtered_mcs_id_list: List of MCS ID arrays with only main MCS IDs.
    """
    filtered_mcs_id_list = []
    for mcs_id_array in mcs_id_list:
        # Create a copy to avoid modifying the original array
        filtered_array = mcs_id_array.copy()
        # Set IDs not in main_mcs_ids to zero
        mask = ~np.isin(filtered_array, main_mcs_ids)
        filtered_array[mask] = 0
        filtered_mcs_id_list.append(filtered_array)
    return filtered_mcs_id_list


def track_mcs(detection_results, main_lifetime_thresh=6):
    """
    Track MCSs across time steps using detection results.
     Returns mcs_detected_list, mcs_id_list, lifetime_list, time_list, lat, lon, main_mcs_ids.
    """
    previous_labeled_regions = None
    previous_cluster_ids = {}
    next_cluster_id = 1

    mcs_detected_list = []
    mcs_id_list = []
    time_list = []
    lifetime_list = []
    lat = None
    lon = None

    # Dictionary to keep track of the lifetime for each MCS ID
    lifetime_dict = {}

    for detection_result in detection_results:
        final_labeled_regions = detection_result['final_labeled_regions']
        current_time = detection_result['time']
        current_lat = detection_result['lat']
        current_lon = detection_result['lon']

        if lat is None:
            lat = current_lat
            lon = current_lon

        mcs_detected = np.zeros_like(final_labeled_regions, dtype=np.int8)
        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)
        
        current_cluster_ids = {}

        cluster_labels = np.unique(final_labeled_regions)
        cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background

        for label in cluster_labels:
            cluster_mask = final_labeled_regions == label
            mcs_detected[cluster_mask] = 1

            if previous_labeled_regions is None:
                # First time step, assign new IDs
                mcs_id[cluster_mask] = next_cluster_id
                current_cluster_ids[label] = next_cluster_id
                # Initialize lifetime
                lifetime_dict[next_cluster_id] = 1
                mcs_lifetime[cluster_mask] = lifetime_dict[next_cluster_id]
                next_cluster_id += 1
            else:
                # Compare with previous clusters
                overlap_area = {}
                for prev_label, prev_id in previous_cluster_ids.items():
                    prev_cluster_mask = previous_labeled_regions == prev_label
                    overlap = np.logical_and(cluster_mask, prev_cluster_mask)
                    overlap_cells = np.sum(overlap)
                    if overlap_cells > 0:
                        # Calculate overlap percentage relative to current cluster
                        current_cluster_area = np.sum(cluster_mask)
                        overlap_percentage = (overlap_cells / current_cluster_area) * 100
                        if overlap_percentage >= 10:  # 10% threshold overlap to merge  # TODO: put this in a config file
                            overlap_area[prev_id] = overlap_percentage

                if len(overlap_area) == 1:
                    # Single overlap, assign existing ID
                    assigned_id = list(overlap_area.keys())[0]
                    mcs_id[cluster_mask] = assigned_id
                    current_cluster_ids[label] = assigned_id
                    # Update lifetime
                    lifetime_dict[assigned_id] += 1
                    mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                elif len(overlap_area) == 0:
                    # New initation, assign new ID and lifetime
                    mcs_id[cluster_mask] = next_cluster_id
                    current_cluster_ids[label] = next_cluster_id
                    # Initialize lifetime
                    lifetime_dict[next_cluster_id] = 1
                    mcs_lifetime[cluster_mask] = lifetime_dict[next_cluster_id]
                    next_cluster_id += 1
                else:
                    # Splitting or merging event
                    prev_ids = list(overlap_area.keys())
                    if len(prev_ids) == 1:
                        # Splitting event
                        assigned_id = prev_ids[0]
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id
                        # Update lifetime
                        lifetime_dict[assigned_id] += 1
                        mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                    else:
                        # Merging event
                        # Assign the ID of the cluster with the largest overlap
                        assigned_id = max(overlap_area, key=overlap_area.get)
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id
                        # Update lifetime
                        lifetime_dict[assigned_id] += 1
                        mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]

        # Update previous clusters
        previous_labeled_regions = final_labeled_regions.copy()
        previous_cluster_ids = current_cluster_ids.copy()

        # Append results
        mcs_detected_list.append(mcs_detected)
        mcs_id_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        # Calculate total lifetime for each MCS ID
        total_lifetime_dict = {}

        for mcs_id_array in mcs_id_list:
            unique_ids = np.unique(mcs_id_array)
            unique_ids = unique_ids[unique_ids != 0]  # Exclude background
            for uid in unique_ids:
                total_lifetime_dict[uid] = total_lifetime_dict.get(uid, 0) + 1

        # Identify main MCS IDs based on lifetime threshold
        main_mcs_ids = [uid for uid, life in total_lifetime_dict.items() if life >= main_lifetime_thresh]


    return mcs_detected_list, mcs_id_list, lifetime_list, time_list, lat, lon, main_mcs_ids
