import numpy as np
import warnings

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
        current_prec = detection_result['prec']
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
                    warnings.warn(f"Cluster splitting or merging detected at time {current_time}.")
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
