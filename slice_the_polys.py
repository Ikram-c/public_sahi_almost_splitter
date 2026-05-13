import networkx as nx
import itertools
import numpy as np
from collections import defaultdict
import json
import tempfile
import os
from pycocotools.coco import COCO
from scipy.interpolate import interp1d
from enum import IntEnum
from typing import Optional


class Axis(IntEnum):
    """Enumeration for axis selection."""
    X = 0
    Y = 1


class SingleSlice:
    """
    Represents a single slice along an axis with threshold-based binarization
    and interpolation at transition points.

    Attributes:
        axis: The axis along which the slice is taken (X or Y)
        axis_vector: Original axis values
        threshold: Threshold value for binarization
        binary_vector: Binarized version of axis_vector
        transition_indices: Indices where binary transitions occur
        interpolator: Interpolation function for transition points
    """

    def __init__(
            self,
            axis_vector: np.ndarray,
            threshold: float,
            axis: Axis = Axis.X
    ):
        """
        Initialize a SingleSlice instance.

        Args:
            axis_vector: 1D array of values along the axis
            threshold: Threshold value for binarization
            axis: Axis identifier (0 for X, 1 for Y)

        Raises:
            ValueError: If axis_vector is empty or not 1D
        """
        if axis_vector.size == 0:
            raise ValueError("axis_vector cannot be empty")
        if axis_vector.ndim != 1:
            raise ValueError("axis_vector must be 1-dimensional")

        self.axis = Axis(axis)
        self.axis_vector = axis_vector.copy()  # Store original values
        self.threshold = threshold
        self.binary_vector = binarize_array(axis_vector, threshold)
        self.transition_indices = find_transitions(self.binary_vector)
        self.interpolator = self._create_interpolator()

    def _create_interpolator(self) -> Optional[interp1d]:
        """
        Create an interpolation function for transition points.

        Returns:
            Interpolation function, or None if insufficient transition points
        """
        if len(self.transition_indices) < 2:
            return None

        return interp1d(
            self.transition_indices,
            self.axis_vector[self.transition_indices],
            bounds_error=False,
            fill_value='extrapolate'
        )

    def interpolate_at(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """
        Interpolate values at given indices using transition points.

        Args:
            indices: Array of indices at which to interpolate

        Returns:
            Interpolated values, or None if interpolator unavailable
        """
        if self.interpolator is None:
            return None
        return self.interpolator(indices)


def binarize_array(
        array: np.ndarray,
        threshold: float,
        copy: bool = True
) -> np.ndarray:
    """
    Binarize an array based on a threshold value.

    Values below threshold become 0, values above become 1.
    Values equal to threshold remain unchanged (handled as boundary case).

    Args:
        array: Input array to binarize
        threshold: Threshold value for binarization
        copy: If True, create a copy; if False, modify in-place

    Returns:
        Binarized array (0s and 1s)
    """
    if copy:
        array = array.copy()

    # Handle the three cases explicitly
    array[array < threshold] = 0
    array[array > threshold] = 1
    # Values equal to threshold can be treated as 0 or 1 depending on use case
    # Here we'll treat them as 1 (upper boundary inclusive)
    array[array == threshold] = 1

    return array


def find_transitions(binary_array: np.ndarray) -> np.ndarray:
    """
    Find indices where binary transitions (0->1 or 1->0) occur.

    This function uses NumPy vectorization for efficiency with large arrays.

    Args:
        binary_array: 1D NumPy array containing only 0s and 1s

    Returns:
        1D array of indices where transitions occur (index of the new value)

    Raises:
        ValueError: If input is not 1D or contains non-binary values

    Example:
        >>> arr = np.array([0, 0, 1, 1, 0, 1])
        >>> find_transitions(arr)
        array([2, 4, 5])
    """
    if binary_array.ndim != 1:
        raise ValueError("Input must be 1-dimensional")

    if binary_array.size == 0:
        return np.array([], dtype=int)

    # Calculate differences between adjacent elements
    # 0->1 becomes 1, 1->0 becomes -1, no change becomes 0
    diff_array = np.diff(binary_array)

    # Find indices where transitions occur
    # Add 1 to get the index where the new value begins
    transition_indices = np.where(diff_array != 0)[0] + 1

    return transition_indices


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

    # Check which boundaries are crossed
    crosses_left = rect_b[0] < rect_a[0] and rect_b[2] > rect_a[0]
    crosses_right = rect_b[0] < rect_a[2] and rect_b[2] > rect_a[2]
    crosses_top = rect_b[1] < rect_a[1] and rect_b[3] > rect_a[1]
    crosses_bottom = rect_b[1] < rect_a[3] and rect_b[3] > rect_a[3]

    # Collect which boundaries are crossed
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

    # Determine overall type
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


def split_polygon_by_boundary(polygon, threshold, axis):
    """
    Split a polygon along a boundary using the SingleSlice class.

    Args:
        polygon: COCO segmentation polygon (list of coordinates [x1, y1, x2, y2, ...])
        threshold: The boundary value to split at
        axis: Axis to split along (Axis.X or Axis.Y)

    Returns:
        dict: Contains split information including transition indices and interpolated values
    """
    if not polygon or len(polygon) < 6:  # Need at least 3 points (6 values)
        return None

    # Convert flat list to coordinate pairs
    coords = np.array(polygon).reshape(-1, 2)

    # Extract the axis values
    if axis == Axis.X:
        axis_values = coords[:, 0]
    else:  # Axis.Y
        axis_values = coords[:, 1]

    # Create SingleSlice instance
    try:
        slice_obj = SingleSlice(axis_values, threshold, axis)

        # Get transition information
        result = {
            'axis': 'X' if axis == Axis.X else 'Y',
            'threshold': threshold,
            'transition_indices': slice_obj.transition_indices.tolist(),
            'binary_vector': slice_obj.binary_vector.tolist(),
            'original_coords': coords.tolist()
        }

        # Add interpolated values if available
        if slice_obj.interpolator is not None and len(slice_obj.transition_indices) > 0:
            interpolated = slice_obj.interpolate_at(slice_obj.transition_indices)
            if interpolated is not None:
                result['interpolated_values'] = interpolated.tolist()

        return result

    except (ValueError, Exception) as e:
        print(f"Error splitting polygon: {e}")
        return None


def process_segmentation_splits(polygon, boundaries_info, rect_a):
    """
    Process a segmentation polygon by splitting it along all crossed boundaries.

    Args:
        polygon: COCO segmentation polygon
        boundaries_info: Dictionary with horizontal_boundaries and vertical_boundaries lists
        rect_a: The rectangle A being crossed [x1, y1, x2, y2]

    Returns:
        dict: Split information for each boundary crossed
    """
    splits = {}

    # Process vertical boundaries (left/right)
    for boundary in boundaries_info['vertical_boundaries']:
        if boundary == 'left':
            threshold = rect_a[0]
            split_result = split_polygon_by_boundary(polygon, threshold, Axis.X)
            if split_result:
                splits['left'] = split_result
        elif boundary == 'right':
            threshold = rect_a[2]
            split_result = split_polygon_by_boundary(polygon, threshold, Axis.X)
            if split_result:
                splits['right'] = split_result

    # Process horizontal boundaries (top/bottom)
    for boundary in boundaries_info['horizontal_boundaries']:
        if boundary == 'top':
            threshold = rect_a[1]
            split_result = split_polygon_by_boundary(polygon, threshold, Axis.Y)
            if split_result:
                splits['top'] = split_result
        elif boundary == 'bottom':
            threshold = rect_a[3]
            split_result = split_polygon_by_boundary(polygon, threshold, Axis.Y)
            if split_result:
                splits['bottom'] = split_result

    return splits if splits else None


def split_polygon_coordinates(polygon, split_info, rect_a):
    """
    Generate new polygon coordinates after splitting along boundaries.

    Args:
        polygon: Original COCO polygon [x1, y1, x2, y2, ...]
        split_info: Split information from process_segmentation_splits
        rect_a: The rectangle A being crossed [x1, y1, x2, y2]

    Returns:
        dict: Dictionary with region names as keys and polygon segments as values
              e.g., {'inside': [...], 'outside': [...]}
    """
    if not polygon or not split_info:
        return {'original': polygon}

    coords = np.array(polygon).reshape(-1, 2)
    segments = {}

    # For now, return a simple split based on the first boundary
    # A more sophisticated implementation would handle multiple boundaries
    boundaries = list(split_info.keys())
    if not boundaries:
        return {'original': polygon}

    first_boundary = boundaries[0]
    boundary_data = split_info[first_boundary]

    # Get the threshold and axis
    threshold = boundary_data['threshold']
    axis_name = boundary_data['axis']
    axis_idx = 0 if axis_name == 'X' else 1

    # Split coordinates based on threshold
    inside_points = []
    outside_points = []

    for i, point in enumerate(coords):
        if first_boundary in ['left', 'top']:
            # For left/top boundaries, inside means >= threshold
            if point[axis_idx] >= threshold:
                inside_points.append(point)
            else:
                outside_points.append(point)
        else:  # right/bottom
            # For right/bottom boundaries, inside means <= threshold
            if point[axis_idx] <= threshold:
                inside_points.append(point)
            else:
                outside_points.append(point)

    # Add transition points at the boundary
    transition_indices = boundary_data.get('transition_indices', [])
    if transition_indices and 'interpolated_values' in boundary_data:
        # Add interpolated boundary points
        for idx, interp_val in zip(transition_indices, boundary_data['interpolated_values']):
            if idx < len(coords):
                # Create a point at the boundary
                if axis_idx == 0:  # X axis
                    boundary_point = [threshold, interp_val]
                else:  # Y axis
                    boundary_point = [interp_val, threshold]

                # Add to both segments for continuity
                if inside_points:
                    inside_points.append(boundary_point)
                if outside_points:
                    outside_points.append(boundary_point)

    # Convert back to flat list format
    if len(inside_points) >= 3:
        segments['inside'] = np.array(inside_points).flatten().tolist()
    if len(outside_points) >= 3:
        segments['outside'] = np.array(outside_points).flatten().tolist()

    # If no valid segments, return original
    if not segments:
        segments['original'] = polygon

    return segments


def calculate_bbox_from_polygon(polygon):
    """
    Calculate bounding box from polygon coordinates.

    Args:
        polygon: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        list: Bounding box in COCO format [x, y, width, height]
    """
    if not polygon or len(polygon) < 6:
        return [0, 0, 0, 0]

    coords = np.array(polygon).reshape(-1, 2)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def calculate_polygon_area(polygon):
    """
    Calculate area of a polygon using the shoelace formula.

    Args:
        polygon: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        float: Area of the polygon
    """
    if not polygon or len(polygon) < 6:
        return 0.0

    coords = np.array(polygon).reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)


def analyze_b_intersections(rectangles_a, rectangles_b, b_idx_to_ann_id=None, b_idx_to_segmentation=None):
    """
    Analyze intersections from rectangle B's perspective.

    For each rectangle in B, find which rectangles in A it intersects with,
    excluding cases where B is completely contained within A.
    If segmentation data is provided, split polygons along crossed boundaries.

    Args:
        rectangles_a: List of rectangles in format [x1, y1, x2, y2]
        rectangles_b: List of rectangles in format [x1, y1, x2, y2]
        b_idx_to_ann_id: Optional mapping from B indices to annotation IDs
        b_idx_to_segmentation: Optional mapping from B indices to segmentation polygons

    Returns:
        dict: For each B rectangle (index or ID), lists the A rectangles it intersects
              and the type of intersection ('horizontal', 'vertical', or 'both')
              and polygon splits if segmentation data is provided
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
                intersection_entry = {
                    'rect_a_index': a_idx,
                    'intersection_type': intersection_info['type'],
                    'boundaries_crossed': intersection_info['boundaries_crossed'],
                    'horizontal_boundaries': intersection_info['horizontal_boundaries'],
                    'vertical_boundaries': intersection_info['vertical_boundaries']
                }

                # Add polygon splits if segmentation data is available
                if b_idx_to_segmentation and b_idx in b_idx_to_segmentation:
                    segmentation = b_idx_to_segmentation[b_idx]
                    # COCO segmentation can be a list of polygons (for complex shapes)
                    # Process the first polygon for simplicity
                    if segmentation and len(segmentation) > 0:
                        polygon = segmentation[0]
                        splits = process_segmentation_splits(polygon, intersection_info, rect_a)
                        if splits:
                            intersection_entry['polygon_splits'] = splits

                intersections.append(intersection_entry)

        # Only include if there are actual intersections (not contained)
        if intersections:
            b_key = b_idx_to_ann_id[b_idx] if b_idx_to_ann_id else b_idx
            result[b_key] = {
                'intersecting_rectangles_a': intersections,
                'intersection_count': len(intersections)
            }

    return result


def update_coco_with_splits(coco_data, intersection_results, rectangles_a,
                            b_idx_to_ann_id, b_idx_to_segmentation):
    """
    Update COCO JSON data with split annotations.

    Args:
        coco_data: Original COCO data dictionary
        intersection_results: Results from analyze_b_intersections
        rectangles_a: List of rectangles A used for splitting
        b_idx_to_ann_id: Mapping from B indices to annotation IDs
        b_idx_to_segmentation: Mapping from B indices to segmentation polygons

    Returns:
        dict: Updated COCO data with split annotations
    """
    # Create a copy of the COCO data
    updated_coco = {
        'images': coco_data['images'].copy(),
        'categories': coco_data['categories'].copy(),
        'annotations': []
    }

    # Reverse mapping: ann_id to b_idx
    ann_id_to_b_idx = {v: k for k, v in b_idx_to_ann_id.items()}

    # Find max annotation ID to generate new IDs
    max_ann_id = max([ann['id'] for ann in coco_data['annotations']])
    next_ann_id = max_ann_id + 1

    # Track which annotations were split
    split_ann_ids = set()

    # Create new annotations for split polygons
    new_annotations = []

    for ann_id, intersection_data in intersection_results.items():
        split_ann_ids.add(ann_id)

        # Get the original annotation
        original_ann = None
        for ann in coco_data['annotations']:
            if ann['id'] == ann_id:
                original_ann = ann.copy()
                break

        if not original_ann:
            continue

        # Get b_idx
        b_idx = ann_id_to_b_idx.get(ann_id)
        if b_idx is None or b_idx not in b_idx_to_segmentation:
            continue

        # Process each intersection
        for intersection in intersection_data['intersecting_rectangles_a']:
            if 'polygon_splits' not in intersection:
                continue

            rect_a_idx = intersection['rect_a_index']
            rect_a = rectangles_a[rect_a_idx]

            # Get the original polygon (first one if multiple)
            original_polygon = b_idx_to_segmentation[b_idx][0]

            # Split the polygon
            segments = split_polygon_coordinates(
                original_polygon,
                intersection['polygon_splits'],
                rect_a
            )

            # Create new annotations for each segment
            for segment_name, segment_polygon in segments.items():
                new_ann = original_ann.copy()
                new_ann['id'] = next_ann_id
                next_ann_id += 1

                # Update segmentation
                new_ann['segmentation'] = [segment_polygon]

                # Recalculate bbox
                new_ann['bbox'] = calculate_bbox_from_polygon(segment_polygon)

                # Recalculate area
                new_ann['area'] = calculate_polygon_area(segment_polygon)

                # Add metadata about the split
                new_ann['split_from'] = ann_id
                new_ann['split_segment'] = segment_name
                new_ann['split_by_rect_a'] = rect_a_idx

                new_annotations.append(new_ann)

    # Add non-split annotations as-is
    for ann in coco_data['annotations']:
        if ann['id'] not in split_ann_ids:
            updated_coco['annotations'].append(ann)

    # Add new split annotations
    updated_coco['annotations'].extend(new_annotations)

    return updated_coco


def save_split_coco(coco_data, intersection_results, rectangles_a,
                    b_idx_to_ann_id, b_idx_to_segmentation, output_path):
    """
    Save the split COCO data to a file.

    Args:
        coco_data: Original COCO data dictionary
        intersection_results: Results from analyze_b_intersections
        rectangles_a: List of rectangles A used for splitting
        b_idx_to_ann_id: Mapping from B indices to annotation IDs
        b_idx_to_segmentation: Mapping from B indices to segmentation polygons
        output_path: Path to save the updated COCO JSON

    Returns:
        dict: Updated COCO data
    """
    updated_coco = update_coco_with_splits(
        coco_data,
        intersection_results,
        rectangles_a,
        b_idx_to_ann_id,
        b_idx_to_segmentation
    )

    with open(output_path, 'w') as f:
        json.dump(updated_coco, f, indent=2)

    print(f"Split COCO data saved to: {output_path}")
    print(f"Original annotations: {len(coco_data['annotations'])}")
    print(f"Updated annotations: {len(updated_coco['annotations'])}")

    return updated_coco


def load_coco_data_for_processing(coco_api, img_ids):
    """
    Loads bounding boxes, segmentations, and creates an index-to-ID map from a COCO object.

    Args:
        coco_api (COCO): An initialized pycocotools COCO object.
        img_ids (list): A list of image IDs to load annotations for.

    Returns:
        tuple: (rectangles_b, b_idx_to_ann_id, b_idx_to_segmentation)
            - rectangles_b: A list of bounding boxes in [x1, y1, x2, y2] format.
            - b_idx_to_ann_id: A dictionary mapping list index to COCO annotation ID.
            - b_idx_to_segmentation: A dictionary mapping list index to segmentation polygon.
    """
    rectangles_b = []
    b_idx_to_ann_id = {}
    b_idx_to_segmentation = {}
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

            # Store segmentation if available
            if 'segmentation' in ann and ann['segmentation']:
                b_idx_to_segmentation[current_idx] = ann['segmentation']

            current_idx += 1

    return rectangles_b, b_idx_to_ann_id, b_idx_to_segmentation


# --- Example usage ---
if __name__ == "__main__":

    # 1. Create a dummy COCO annotation file with various intersection scenarios
    # Include segmentation polygons for testing
    dummy_coco_data = {
        "images": [
            {"id": 1, "width": 200, "height": 200}
        ],
        "annotations": [
            # Completely contained - should be excluded
            {
                "id": 101,
                "image_id": 1,
                "bbox": [60, 60, 20, 20],
                "category_id": 1,
                "iscrowd": 0,
                "area": 400,
                "segmentation": [[60, 60, 80, 60, 80, 80, 60, 80]]
            },
            # Crosses left boundary (vertical intersection)
            {
                "id": 102,
                "image_id": 1,
                "bbox": [30, 60, 30, 20],
                "category_id": 1,
                "iscrowd": 0,
                "area": 600,
                "segmentation": [[30, 60, 60, 60, 60, 80, 30, 80]]
            },
            # Crosses top boundary (horizontal intersection)
            {
                "id": 103,
                "image_id": 1,
                "bbox": [60, 30, 20, 30],
                "category_id": 1,
                "iscrowd": 0,
                "area": 600,
                "segmentation": [[60, 30, 80, 30, 80, 60, 60, 60]]
            },
            # Crosses top-left corner (both)
            {
                "id": 104,
                "image_id": 1,
                "bbox": [30, 30, 30, 30],
                "category_id": 1,
                "iscrowd": 0,
                "area": 900,
                "segmentation": [[30, 30, 60, 30, 60, 60, 30, 60]]
            },
            # No intersection
            {
                "id": 105,
                "image_id": 1,
                "bbox": [150, 150, 20, 20],
                "category_id": 1,
                "iscrowd": 0,
                "area": 400,
                "segmentation": [[150, 150, 170, 150, 170, 170, 150, 170]]
            },
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

        # 3. Load data for images [1] including segmentations
        img_ids_to_process = [1]
        rectangles_b, b_idx_to_ann_id, b_idx_to_segmentation = load_coco_data_for_processing(
            coco_api,
            img_ids_to_process
        )

        print(f"Loaded {len(rectangles_b)} annotations from COCO.")
        print(f"Rectangles B: {rectangles_b}")
        print(f"Segmentations available: {len(b_idx_to_segmentation)}\n")

        # 4. Define rectangles_a (the "slicing" rectangles)
        # Define a region from (50, 50) to (100, 100)
        rectangles_a = [
            [50, 50, 100, 100]
        ]

        print(f"Rectangles A: {rectangles_a}\n")

        # 5. Analyze intersections from B's perspective with segmentation splitting
        intersection_results = analyze_b_intersections(
            rectangles_a,
            rectangles_b,
            b_idx_to_ann_id,
            b_idx_to_segmentation  # Pass segmentation data
        )

        print("=" * 80)
        print("--- Intersection Analysis (B's Perspective with Polygon Splitting) ---")
        print("=" * 80)
        print("\nFor each rectangle in B (COCO annotations), shows:")
        print("  - Which rectangles in A it intersects with")
        print("  - Type of intersection (horizontal/vertical/both)")
        print("  - Specific boundaries crossed (left/right/top/bottom)")
        print("  - Polygon split information (transition points, binary vector)")
        print("  - Excludes rectangles that are completely contained\n")
        print(json.dumps(intersection_results, indent=2))

        # 6. Load original COCO data and create split version
        print("\n" + "=" * 80)
        print("--- Creating Split COCO JSON ---")
        print("=" * 80 + "\n")

        with open(tmp_ann_file, 'r') as f:
            original_coco_data = json.load(f)

        # Create output path for split COCO
        split_output_path = tmp_ann_file.replace('.json', '_split.json')

        # Save the split COCO data
        updated_coco_data = save_split_coco(
            original_coco_data,
            intersection_results,
            rectangles_a,
            b_idx_to_ann_id,
            b_idx_to_segmentation,
            split_output_path
        )

        # Display some of the split annotations
        print("\n--- Sample Split Annotations ---")
        for ann in updated_coco_data['annotations'][:5]:
            print(f"\nAnnotation ID: {ann['id']}")
            if 'split_from' in ann:
                print(f"  Split from: {ann['split_from']}")
                print(f"  Segment: {ann['split_segment']}")
                print(f"  Split by rect_a: {ann['split_by_rect_a']}")
            print(f"  BBox: {ann['bbox']}")
            print(f"  Area: {ann['area']:.2f}")
            print(f"  Segmentation points: {len(ann['segmentation'][0]) // 2}")

        # Clean up the split file
        if os.path.exists(split_output_path):
            os.remove(split_output_path)
            print(f"\n(Cleaned up temporary split file: {split_output_path})")

    finally:
        # Clean up the temporary file
        os.remove(tmp_ann_file)