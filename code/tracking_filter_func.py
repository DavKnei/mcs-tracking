import numpy as np


def filter_main_mcs(mcs_ids_list, main_mcs_ids):
    """
    Filters tracking results to include only the 'main' MCS IDs.

    Args:
        mcs_ids_list (List[numpy.ndarray]): List of 2D arrays with track IDs.
        main_mcs_ids (List[int]): List of track IDs considered 'main' MCS.

    Returns:
        List[numpy.ndarray]: A new list of arrays, where IDs not in main_mcs_ids are set to 0.
    """
    filtered_mcs_id_list = []
    for mcs_id_array in mcs_ids_list:
        filtered_array = mcs_id_array.copy()
        mask = ~np.isin(filtered_array, main_mcs_ids)
        filtered_array[mask] = 0
        filtered_mcs_id_list.append(filtered_array)
    return filtered_mcs_id_list


def filter_relevant_systems(
    mcs_ids_list, main_mcs_ids, merging_events, splitting_events
):
    """
    Filters the tracking results to include only relevant track IDs. Main MCSs and systems that merge into
    MCSs or split of MCSs.

    This function computes the union of:
      - main_mcs_ids (the main MCS tracks that satisfy the area–lifetime criteria),
      - All track IDs involved in merging events (both the parent IDs and the child ID), and
      - All track IDs involved in splitting events (both the parent ID and all child IDs).

    It then processes the list of MCS ID arrays (mcs_ids_list) such that any pixel value
    that is not in this union is set to 0.

    Args:
        mcs_ids_list (List[np.ndarray]): List of 2D arrays (one per timestep) with track IDs.
        main_mcs_ids (List[int]): List of track IDs that satisfy the main MCS criteria.
        merging_events (List[MergingEvent]): List of merging events recorded during tracking.
        splitting_events (List[SplittingEvent]): List of splitting events recorded during tracking.

    Returns:
        List[np.ndarray]: A new list of 2D arrays where any pixel with a track ID not in the
                          union of relevant IDs is set to 0.

    Usage:
        After running track_mcs(), suppose you obtain:

            mcs_ids_list, main_mcs_ids, lifetime_list, time_list, lat, lon,
            merging_events, splitting_events, tracking_centers_list = track_mcs(...)

        Then, to filter out ephemeral tracks that never merged or split into a main MCS, do:

            filtered_mcs_ids_list = filter_relevant_mcs(
                mcs_ids_list, main_mcs_ids, merging_events, splitting_events
            )
    """
    # Start with the main MCS IDs.
    relevant_ids = set(main_mcs_ids)

    # Add IDs involved in merging events of MCSs.
    for event in merging_events:
        if event.child_id in relevant_ids:
            relevant_ids.update(event.parent_ids)
            relevant_ids.add(event.child_id)

    # Add IDs involved in splitting events of MCSs.
    for event in splitting_events:
        if event.parent_id in relevant_ids:
            relevant_ids.add(event.parent_id)
            relevant_ids.update(event.child_ids)

    # Filter each timestep's array: only keep values in relevant_ids.
    filtered_mcs_ids_list = []
    for mcs_array in mcs_ids_list:
        filtered_array = mcs_array.copy()
        # Any pixel not in relevant_ids is set to 0.
        mask = ~np.isin(filtered_array, list(relevant_ids))
        filtered_array[mask] = 0
        filtered_mcs_ids_list.append(filtered_array)

    return filtered_mcs_ids_list


def apply_li_filter(
    mcs_ids_list, lifting_index_regions_list, time_list, main_lifetime_thresh
):
    """
    Post‑filter MCS tracks by lifting_index_regions (0/1).
    A track passes if any LI==1 occurs within two hours before or at its start,
    or if LI==1 first appears later and the remaining track length ≥ main_lifetime_thresh.

    Returns a new list of 2D arrays (same shape as mcs_ids_list) called main_mcs_id_robust.
    """
    robust_ids = [np.zeros_like(arr, dtype=int) for arr in mcs_ids_list]
    track_ids = np.unique(
        np.concatenate([arr[arr > 0].ravel() for arr in mcs_ids_list])
    )
    track_ids = track_ids[track_ids != 0]

    # Build per-track time indices
    for tid in track_ids:
        # Find all time steps where this track exists
        times_present = [i for i, arr in enumerate(mcs_ids_list) if (arr == tid).any()]

        if not times_present:
            continue
        t0 = times_present[0]
        # Define pre-window (two hours before t0)
        window = list(range(max(0, t0 - 2), t0 + 1))

        # Check for any convective LI in that window
        found = False
        for ti in window:
            mask = mcs_ids_list[ti] == tid
            if mask.any() and (lifting_index_regions_list[ti][mask] == 1).any():
                onset = t0
                found = True
                break

        # If not found, look for first later LI==1
        if not found:
            for ti in times_present:
                mask = mcs_ids_list[ti] == tid
                if mask.any() and (lifting_index_regions_list[ti][mask] == 1).any():
                    onset = ti
                    found = True
                    break

        if not found:
            # No LI==1 anywhere → discard entire track
            continue

        # Ensure remaining lifetime ≥ threshold
        remaining = len([i for i in times_present if i >= onset])
        if remaining < main_lifetime_thresh:
            continue

        # Mark robust IDs from onset onward
        for ti in times_present:
            if ti >= onset:
                robust_ids[ti][mcs_ids_list[ti] == tid] = tid

    return robust_ids
