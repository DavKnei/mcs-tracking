import numpy as np
import datetime
from dataclasses import dataclass
from typing import List
from tracking_helper_func import get_dominant_cluster


@dataclass
class MergingEvent:
    """Stores information about a merging event in the tracking."""

    time: datetime.datetime
    parent_ids: List[int]
    child_id: int
    parent_areas: List[float]
    child_area: float


def handle_merging(
    new_label,
    old_track_ids,
    merging_events,
    final_labeled_regions,
    current_time,
    max_area_dict,
    grid_cell_area_km2,
    nmaxmerge=5,
):
    """
    Resolves a merging scenario where multiple old track IDs map to a single new label.

    Args:
        new_label (int): The new detection label that overlaps multiple old track IDs.
        old_track_ids (List[int]): The list of old track IDs that overlap this new label.
        merging_events (List[MergingEvent]): A list to record merging events.
        final_labeled_regions (numpy.ndarray): Current labeled regions.
        current_time (datetime.datetime): Timestamp for the merging event.
        max_area_dict (dict): Maps track ID -> max area encountered.
        grid_cell_area_km2 (float): Factor to convert pixel count to km².
        nmaxmerge (int, optional): Max number of merges recorded. Defaults to 5.

    Returns:
        int: The 'dominant' old track ID that this new label will inherit.
    """
    if len(old_track_ids) == 1:
        return old_track_ids[0]

    best_id = get_dominant_cluster(old_track_ids, max_area_dict)

    if len(old_track_ids) > nmaxmerge:
        old_track_ids = old_track_ids[:nmaxmerge]

    if merging_events is not None and len(old_track_ids) > 1:
        mask = final_labeled_regions == new_label
        area_pix = np.sum(mask)
        child_area = area_pix * grid_cell_area_km2
        parent_areas = [max_area_dict[pid] for pid in old_track_ids]
        mevt = MergingEvent(
            time=current_time,
            parent_ids=old_track_ids,
            child_id=best_id,
            parent_areas=parent_areas,
            child_area=child_area,
        )
        merging_events.append(mevt)

    return best_id
