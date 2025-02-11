import numpy as np
import logging
from collections import defaultdict
from tracking_filter_func import filter_relevant_systems
from tracking_helper_func import (
    assign_new_id,
    check_overlaps,
    handle_continuation,
    handle_no_overlap,
)
from tracking_merging import handle_merging
from tracking_splitting import handle_splitting


def track_mcs(
    detection_results,
    main_lifetime_thresh,
    main_area_thresh,
    grid_cell_area_km2,
    nmaxmerge,
):
    """
    Tracks MCSs across multiple timesteps using spatial overlap and stable ID assignment.

    Args:
        detection_results (List[dict]): Each dict contains:
            "final_labeled_regions" (2D array of labels),
            "center_points" (dict mapping label->(lat, lon)), optional
            "time" (datetime.datetime),
            "lat" (2D array),
            "lon" (2D array).
        main_lifetime_thresh (int): Minimum consecutive lifetime (in timesteps) for main MCS.
        main_area_thresh (float): Minimum area (km²) that must be reached in each timestep.
        grid_cell_area_km2 (float): Factor to convert pixel count to area (km²).
        nmaxmerge (int): Max allowed merging/splitting in a single timestep.

    Returns:
        Tuple:
            mcs_ids_list (List[numpy.ndarray]): Track ID arrays per timestep.
            main_mcs_ids (List[int]): List of IDs considered main MCS by end of tracking.
            lifetime_list (List[numpy.ndarray]): Per-timestep 2D arrays for pixel-wise lifetime.
            time_list (List[datetime.datetime]): Timestamps for each timestep.
            lat (numpy.ndarray): 2D lat array from the first timestep.
            lon (numpy.ndarray): 2D lon array from the first timestep.
            merging_events (List[MergingEvent]): All recorded merging events.
            splitting_events (List[SplittingEvent]): All recorded splitting events.
            tracking_centers_list (List[dict]): For each timestep, a dict mapping
                track_id -> (center_lat, center_lon).
    """
    logger = logging.getLogger(__name__)

    previous_labeled_regions = None
    previous_cluster_ids = {}
    merge_split_cluster_ids = {}
    next_cluster_id = 1

    mcs_ids_list = []
    lifetime_list = []
    tracking_centers_list = []
    time_list = []
    lat = None
    lon = None

    lifetime_dict = defaultdict(int)
    max_area_dict = defaultdict(float)

    merging_events = []
    splitting_events = []

    for idx, detection_result in enumerate(detection_results):
        final_labeled_regions = detection_result["final_labeled_regions"]
        center_points_dict = detection_result.get("center_points", {})
        current_time = detection_result["time"]
        current_lat = detection_result["lat"]
        current_lon = detection_result["lon"]

        if lat is None:
            lat = current_lat
            lon = current_lon

        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)

        unique_labels = np.unique(final_labeled_regions)
        unique_labels = unique_labels[unique_labels != 0]

        # If no valid clusters are detected, end all tracks for this timestep.
        if len(unique_labels) == 0:
            print(f"No clusters detected at {current_time}")
            previous_cluster_ids = {}
            previous_labeled_regions = None

            mcs_ids_list.append(mcs_id)
            lifetime_list.append(mcs_lifetime)
            time_list.append(current_time)
            tracking_centers_list.append({})
            continue

        # For the first timestep with clusters.
        if previous_labeled_regions is None:
            for label in unique_labels:
                cluster_mask = final_labeled_regions == label
                area = np.sum(cluster_mask) * grid_cell_area_km2
                assigned_id, next_cluster_id = assign_new_id(
                    label,
                    cluster_mask,
                    area,
                    next_cluster_id,
                    lifetime_dict,
                    max_area_dict,
                    mcs_id,
                    mcs_lifetime,
                )
                previous_cluster_ids[label] = assigned_id
                merge_split_cluster_ids[label] = assigned_id
        else:
            # For subsequent timesteps: check overlaps between previous and current clusters.
            overlap_map = check_overlaps(
                previous_labeled_regions,
                final_labeled_regions,
                previous_cluster_ids,
                overlap_threshold=10,
            )
            temp_assigned = {}
            labels_no_overlap = []

            for new_lbl, old_ids in overlap_map.items():
                if len(old_ids) == 0:
                    labels_no_overlap.append(new_lbl)
                elif len(old_ids) == 1:
                    chosen_id = old_ids[0]
                    # Continue the track.
                    handle_continuation(
                        new_label=new_lbl,
                        old_track_id=chosen_id,
                        final_labeled_regions=final_labeled_regions,
                        mcs_id=mcs_id,
                        mcs_lifetime=mcs_lifetime,
                        lifetime_dict=lifetime_dict,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2,
                    )
                    temp_assigned[new_lbl] = chosen_id
                else:
                    # Merging scenario.
                    chosen_id = handle_merging(
                        new_label=new_lbl,
                        old_track_ids=old_ids,
                        merging_events=merging_events,
                        final_labeled_regions=final_labeled_regions,
                        current_time=current_time,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2,
                        nmaxmerge=nmaxmerge,
                    )
                    mask = final_labeled_regions == new_lbl
                    mcs_id[mask] = chosen_id
                    lifetime_dict[chosen_id] += 1
                    temp_assigned[new_lbl] = chosen_id

            # Handle new clusters that had no overlap.
            new_assign_map, next_cluster_id = handle_no_overlap(
                labels_no_overlap,
                final_labeled_regions,
                next_cluster_id,
                lifetime_dict,
                max_area_dict,
                mcs_id,
                mcs_lifetime,
                grid_cell_area_km2,
            )
            temp_assigned.update(new_assign_map)

            # Check for splits: if one old track claims multiple new labels.
            oldid_to_newlist = defaultdict(list)
            for lbl, tid in temp_assigned.items():
                oldid_to_newlist[tid].append(lbl)

            for old_id, newlbls in oldid_to_newlist.items():
                if len(newlbls) > 1:
                    splitted_map, next_cluster_id = handle_splitting(
                        old_id,
                        newlbls,
                        final_labeled_regions,
                        current_time,
                        next_cluster_id,
                        splitting_events,
                        mcs_id,
                        mcs_lifetime,
                        lifetime_dict,
                        max_area_dict,
                        grid_cell_area_km2,
                        nmaxsplit=nmaxmerge,
                    )
                    for nl, finalid in splitted_map.items():
                        temp_assigned[nl] = finalid

            current_cluster_ids = temp_assigned
            previous_cluster_ids = current_cluster_ids
            logger.info(f"MCS tracking of {current_time}.")

        previous_labeled_regions = final_labeled_regions.copy()

        mcs_ids_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        # Build a dictionary of track centers for this timestep.
        centers_this_timestep = {}
        label_by_cluster = defaultdict(list)
        for lbl, tid in previous_cluster_ids.items():
            label_by_cluster[tid].append(lbl)

        for tid, label_list in label_by_cluster.items():
            center_latlon = (None, None)
            for detect_label in label_list:
                detect_label_str = str(detect_label)
                if detect_label_str in center_points_dict:
                    center_latlon = center_points_dict[detect_label_str]
                    break
            centers_this_timestep[str(tid)] = center_latlon
        tracking_centers_list.append(centers_this_timestep)

    # ---- New Filtering Step ----
    # Instead of simply checking overall lifetime and maximum area,
    # we now require that each main MCS has at least main_lifetime_thresh consecutive
    # timesteps with an area >= main_area_thresh.
    #
    # For each track ID, we build a Boolean series for each timestep, where True means
    # the area for that track meets the main_area_thresh. Then we compute the maximum number
    # of consecutive True values. If that maximum is at least main_lifetime_thresh, the track qualifies.

    def compute_max_consecutive(bool_list):
        max_cons = 0
        current = 0
        for b in bool_list:
            if b:
                current += 1
                if current > max_cons:
                    max_cons = current
            else:
                current = 0
        return max_cons

    all_track_ids = list(lifetime_dict.keys())
    valid_ids = []
    for tid in all_track_ids:
        # Build a Boolean series: for each timestep, True if area for track tid >= main_area_thresh.
        bool_series = []
        for mcs_id_array in mcs_ids_list:
            area = np.sum(mcs_id_array == tid) * grid_cell_area_km2
            bool_series.append(area >= main_area_thresh)
        if compute_max_consecutive(bool_series) >= main_lifetime_thresh:
            valid_ids.append(tid)
    main_mcs_ids = valid_ids
    filtered_mcs_ids_list = filter_relevant_systems(
        mcs_ids_list, main_mcs_ids, merging_events, splitting_events
    )
    logger.info(f"Tracking finished. {len(main_mcs_ids)} main MCSs found.")
    return (
        filtered_mcs_ids_list,
        main_mcs_ids,
        lifetime_list,
        time_list,
        lat,
        lon,
        merging_events,
        splitting_events,
        tracking_centers_list,
    )
