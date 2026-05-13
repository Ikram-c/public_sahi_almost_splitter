import networkx as nx
import itertools
import numpy as np
from collections import defaultdict
import json
import tempfile
import os
from pycocotools.coco import COCO


class IntervalUnionQuery:
    """A segment tree implementation for querying the union size of intervals."""

    def __init__(self, L, y_coords):
        assert L != []
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        self.overlaps = []
        self.y_coords = y_coords
        self.sweep_events = []
        self.active_rectangles_a = {}
        self.active_rectangles_b = {}

        for i, val in enumerate(L):
            self.w[self.N + i] = val
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def modify_interval(self, i, k, offset, x_coord, rect_idx, is_from_list_a):
        """Updates coverage counts for the y indices [i, k)."""
        for y in range(i, k):
            if y not in self.active_rectangles_a:
                self.active_rectangles_a[y] = set()
                self.active_rectangles_b[y] = set()

            was_overlapping = bool(self.active_rectangles_a[y]) and bool(self.active_rectangles_b[y])
            old_sets = (set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y]))

            target_set = self.active_rectangles_a[y] if is_from_list_a else self.active_rectangles_b[y]
            if offset == 1:
                target_set.add(rect_idx)
            else:
                target_set.discard(rect_idx)

            is_overlapping = bool(self.active_rectangles_a[y]) and bool(self.active_rectangles_b[y])
            new_sets = (set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y]))

            if was_overlapping != is_overlapping or (is_overlapping and old_sets != new_sets):
                self.sweep_events.append(
                    (x_coord, y, set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y])))

        self._change(1, 0, self.N, i, k, offset)

    def find_overlaps_between_lists(self):
        """Processes sweep events to identify overlap regions."""
        self.sweep_events.sort()
        active_regions = {}
        for x, y, rects_a, rects_b in self.sweep_events:
            if y in active_regions:
                start_x, start_rects_a, start_rects_b = active_regions.pop(y)
                if x > start_x:
                    y_low = self.y_coords[y]
                    y_high = self.y_coords[y + 1]
                    self.overlaps.append((start_x, x, y_low, y_high, start_rects_a, start_rects_b))

            if rects_a and rects_b:
                active_regions[y] = (x, rects_a, rects_b)

    def _change(self, p, start, span, i, k, offset):
        """Recursive helper for segment tree update."""
        if start + span <= i or k <= start: return
        if i <= start and start + span <= k:
            self.c[p] += offset
        else:
            mid = span // 2
            self._change(2 * p, start, mid, i, k, offset)
            self._change(2 * p + 1, start + mid, mid, i, k, offset)

        if self.c[p] == 0:
            self.s[p] = 0 if p >= self.N else self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]


class Event:
    """Represents a start or end event in the sweep-line algorithm."""

    def __init__(self, x, rectangle, is_start, rect_idx, is_from_list_a):
        self.x = x
        self.rectangle = rectangle
        self.is_start = is_start
        self.rect_idx = rect_idx
        self.is_from_list_a = is_from_list_a


def find_overlapping_between_lists(rectangles_a, rectangles_b):
    """Finds which rectangles from list_a overlap with rectangles from list_b."""
    if not rectangles_a or not rectangles_b:
        return None, None, []

    normalized_a = [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])) for r in rectangles_a]
    normalized_b = [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])) for r in rectangles_b]

    events = []
    for i, r in enumerate(normalized_a):
        if r[0] < r[2]:  # Ignore zero-width rectangles
            events.extend([Event(r[0], r, True, i, True), Event(r[2], r, False, i, True)])
    for i, r in enumerate(normalized_b):
        if r[0] < r[2]:  # Ignore zero-width rectangles
            events.extend([Event(r[0], r, True, i, False), Event(r[2], r, False, i, False)])

    if not events:
        return None, None, []

    events.sort(key=lambda e: (e.x, not e.is_start))

    y_coords = sorted(list(set(y for r in normalized_a + normalized_b for y in [r[1], r[3]])))
    if len(y_coords) < 2: return None, None, []

    y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    y_mapping = {val: idx for idx, val in enumerate(y_coords)}

    query = IntervalUnionQuery(y_intervals, y_coords)

    for event in events:
        y0, y1 = y_mapping[event.rectangle[1]], y_mapping[event.rectangle[3]]
        if y0 == y1: continue  # Ignore zero-height rectangles

        offset = 1 if event.is_start else -1
        query.modify_interval(y0, y1, offset, event.x, event.rect_idx, event.is_from_list_a)

    query.find_overlaps_between_lists()

    indices_a, indices_b, pairs = set(), set(), set()
    for _, _, _, _, rects_a, rects_b in query.overlaps:
        indices_a.update(rects_a)
        indices_b.update(rects_b)
        for a_idx in rects_a:
            for b_idx in rects_b:
                pairs.add((a_idx, b_idx))

    return (sorted(list(indices_a)), sorted(list(indices_b)), sorted(list(pairs))) if indices_a else (None, None, [])


def is_contained(rect_b, rect_a):
    """Check if rectangle B is completely contained within rectangle A."""
    return (rect_b[0] >= rect_a[0] and rect_b[2] <= rect_a[2] and
            rect_b[1] >= rect_a[1] and rect_b[3] <= rect_a[3])


def classify_intersection(rect_b, rect_a):
    """
    Classify the type of intersection between rectangle B and rectangle A.

    Returns:
        dict: Contains:
            - 'type': 'contained', 'horizontal', 'vertical', 'both', or 'none'
            - 'boundaries_crossed': list of specific boundaries ('left', 'right', 'top', 'bottom')
            - 'horizontal_boundaries': list of horizontal boundaries crossed
            - 'vertical_boundaries': list of vertical boundaries crossed
    """
    # Check if B is contained in A
    if is_contained(rect_b, rect_a):
        return {
            'type': 'contained',
            'boundaries_crossed': [],
            'horizontal_boundaries': [],
            'vertical_boundaries': []
        }

    # Check for overlap
    if not (rect_b[0] < rect_a[2] and rect_b[2] > rect_a[0] and
            rect_b[1] < rect_a[3] and rect_b[3] > rect_a[1]):
        return {
            'type': 'none',
            'boundaries_crossed': [],
            'horizontal_boundaries': [],
            'vertical_boundaries': []
        }

    crosses_left = rect_b[0] < rect_a[0] and rect_b[2] > rect_a[0]
    crosses_right = rect_b[0] < rect_a[2] and rect_b[2] > rect_a[2]
    crosses_top = rect_b[1] < rect_a[1] and rect_b[3] > rect_a[1]
    crosses_bottom = rect_b[1] < rect_a[3] and rect_b[3] > rect_a[3]

    boundaries_crossed = []
    vertical_boundaries = []
    horizontal_boundaries = []

    if crosses_left:
        boundaries_crossed.append('left')
        vertical_boundaries.append('left')
    if crosses_right:
        boundaries_crossed.append('right')
        vertical_boundaries.append('right')
    if crosses_top:
        boundaries_crossed.append('top')
        horizontal_boundaries.append('top')
    if crosses_bottom:
        boundaries_crossed.append('bottom')
        horizontal_boundaries.append('bottom')

    crosses_vertical = crosses_left or crosses_right
    crosses_horizontal = crosses_top or crosses_bottom

    if crosses_vertical and crosses_horizontal:
        intersection_type = 'both'
    elif crosses_vertical:
        intersection_type = 'vertical'
    elif crosses_horizontal:
        intersection_type = 'horizontal'
    else:
        intersection_type = 'none'

    return {
        'type': intersection_type,
        'boundaries_crossed': boundaries_crossed,
        'horizontal_boundaries': horizontal_boundaries,
        'vertical_boundaries': vertical_boundaries
    }


def analyze_b_intersections(rectangles_a, rectangles_b, b_idx_to_ann_id=None):
    """
    Analyze intersections from rectangle B's perspective.

    For each rectangle in B, find which rectangles in A it intersects with,
    excluding cases where B is completely contained within A.

    Args:
        rectangles_a: List of rectangles in format [x1, y1, x2, y2]
        rectangles_b: List of rectangles in format [x1, y1, x2, y2]
        b_idx_to_ann_id: Optional mapping from B indices to annotation IDs

    Returns:
        dict: For each B rectangle (index or ID), lists the A rectangles it intersects
              and the type of intersection ('horizontal', 'vertical', or 'both')
    """
    _, _, overlap_pairs = find_overlapping_between_lists(rectangles_a, rectangles_b)
    if not overlap_pairs:
        return {}

    # Normalize rectangles
    normalized_a = [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))
                    for r in rectangles_a]
    normalized_b = [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))
                    for r in rectangles_b]

    # Group by B rectangles
    grouped_by_b = defaultdict(list)
    for a_idx, b_idx in overlap_pairs:
        grouped_by_b[b_idx].append(a_idx)

    result = {}
    for b_idx, a_indices in grouped_by_b.items():
        rect_b = normalized_b[b_idx]
        intersections = []

        for a_idx in a_indices:
            rect_a = normalized_a[a_idx]
            intersection_info = classify_intersection(rect_b, rect_a)

            # Exclude contained rectangles
            if intersection_info['type'] not in ['contained', 'none']:
                intersections.append({
                    'rect_a_index': a_idx,
                    'intersection_type': intersection_info['type'],
                    'boundaries_crossed': intersection_info['boundaries_crossed'],
                    'horizontal_boundaries': intersection_info['horizontal_boundaries'],
                    'vertical_boundaries': intersection_info['vertical_boundaries']
                })

        # Only include if there are actual intersections (not contained)
        if intersections:
            b_key = b_idx_to_ann_id[b_idx] if b_idx_to_ann_id else b_idx
            result[b_key] = {
                'intersecting_rectangles_a': intersections,
                'intersection_count': len(intersections)
            }

    return result


def load_coco_data_for_processing(coco_api, img_ids):
    """
    Loads bounding boxes and creates an index-to-ID map from a COCO object.

    Args:
        coco_api (COCO): An initialized pycocotools COCO object.
        img_ids (list): A list of image IDs to load annotations for.

    Returns:
        tuple: (rectangles_b, b_idx_to_ann_id)
            - rectangles_b: A list of bounding boxes in [x1, y1, x2, y2] format.
            - b_idx_to_ann_id: A dictionary mapping list index to COCO annotation ID.
    """
    rectangles_b = []
    b_idx_to_ann_id = {}
    current_idx = 0

    for img_id in img_ids:
        annIds = coco_api.getAnnIds(imgIds=img_id)
        anns = coco_api.loadAnns(annIds)

        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            r = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            rectangles_b.append(r)
            b_idx_to_ann_id[current_idx] = ann['id']
            current_idx += 1

    return rectangles_b, b_idx_to_ann_id


# --- Example usage ---
if __name__ == "__main__":

    # 1. Create a dummy COCO annotation file with various intersection scenarios
    dummy_coco_data = {
        "images": [
            {"id": 1, "width": 200, "height": 200}
        ],
        "annotations": [
            # Completely contained
            {"id": 101, "image_id": 1, "bbox": [60, 60, 20, 20], "category_id": 1, "iscrowd": 0, "area": 400},
            # Crosses left boundary (vertical intersection)
            {"id": 102, "image_id": 1, "bbox": [30, 60, 30, 20], "category_id": 1, "iscrowd": 0, "area": 600},
            # Crosses top boundary (horizontal intersection)
            {"id": 103, "image_id": 1, "bbox": [60, 30, 20, 30], "category_id": 1, "iscrowd": 0, "area": 600},
            # Crosses top-left corner (both)
            {"id": 104, "image_id": 1, "bbox": [30, 30, 30, 30], "category_id": 1, "iscrowd": 0, "area": 900},
            # No intersection
            {"id": 105, "image_id": 1, "bbox": [150, 150, 20, 20], "category_id": 1, "iscrowd": 0, "area": 400},
        ],
        "categories": [
            {"id": 1, "name": "test"}
        ]
    }

    # Use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(dummy_coco_data, f)
        tmp_ann_file = f.name

    try:
        # 2. Initialize COCO API
        coco_api = COCO(tmp_ann_file)

        # 3. Load data for images [1]
        img_ids_to_process = [1]
        rectangles_b, b_idx_to_ann_id = load_coco_data_for_processing(coco_api, img_ids_to_process)

        print(f"Loaded {len(rectangles_b)} annotations from COCO.")
        print(f"Rectangles B: {rectangles_b}\n")

        # 4. Define rectangles_a (the "slicing" rectangles)
        # Define a region from (50, 50) to (100, 100)
        rectangles_a = [
            [50, 50, 100, 100]
        ]

        print(f"Rectangles A: {rectangles_a}\n")

        # 5. Analyze intersections from B's perspective
        intersection_results = analyze_b_intersections(
            rectangles_a,
            rectangles_b,
            b_idx_to_ann_id
        )

        print("--- Intersection Analysis (B's Perspective) ---")
        print("For each rectangle in B (COCO annotations), shows:")
        print("  - Which rectangles in A it intersects with")
        print("  - Type of intersection (horizontal/vertical/both)")
        print("  - Specific boundaries crossed (left/right/top/bottom)")
        print("  - Excludes rectangles that are completely contained\n")
        print(json.dumps(intersection_results, indent=2))

    finally:
        # Clean up the temporary file
        os.remove(tmp_ann_file)