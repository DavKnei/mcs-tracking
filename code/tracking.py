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


def track_mcs(detection_results):
    """
    Track MCSs across time steps using detection results.
    Returns mcs_detected_list, mcs_id_list, time_list, lat, lon.
    """
    previous_labeled_regions = None
    previous_cluster_ids = {}
    next_cluster_id = 1

    mcs_detected_list = []
    mcs_id_list = []
    time_list = []
    lat = None
    lon = None

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
                        if overlap_percentage >= 10:
                            overlap_area[prev_id] = overlap_percentage

                if len(overlap_area) == 1:
                    # Single overlap, assign existing ID
                    assigned_id = list(overlap_area.keys())[0]
                    mcs_id[cluster_mask] = assigned_id
                    current_cluster_ids[label] = assigned_id
                elif len(overlap_area) == 0:
                    # No overlap, assign new ID
                    mcs_id[cluster_mask] = next_cluster_id
                    current_cluster_ids[label] = next_cluster_id
                    next_cluster_id += 1
                else:
                    # Multiple overlaps detected (merging/splitting)
                    prev_id = list(overlap_area.keys())
                    if len(prev_id) == 1:
                        # Splitting event
                        splitting_ids, next_cluster_id = handle_splitting_event(
                            overlap_area, next_cluster_id)
                        mcs_id[cluster_mask] = splitting_ids[label]
                        current_cluster_ids[label] = splitting_ids[label]
                    else:
                    # Merging events
                        warnings.warn(f"Merging not yet implemented but found at time {current_time}.")
                        # For now, assign the ID of the cluster with the largest overlap
                        assigned_id = max(overlap_area, key=overlap_area.get)
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id

        # Update previous clusters
        previous_labeled_regions = final_labeled_regions.copy()
        previous_cluster_ids = current_cluster_ids.copy()

        # Append results
        mcs_detected_list.append(mcs_detected)
        mcs_id_list.append(mcs_id)
        time_list.append(current_time)

    return mcs_detected_list, mcs_id_list, time_list, lat, lon
