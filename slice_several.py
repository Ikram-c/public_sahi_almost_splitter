import networkx as nx
import itertools
import numpy as np
from collections import defaultdict


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


def find_overlapping_with_first(rectangles):
    """Finds which rectangles (indices 1 to n-1) overlap with the first rectangle (index 0)."""
    if not rectangles or len(rectangles) < 2:
        return None
    _, indices_b, _ = find_overlapping_between_lists([rectangles[0]], rectangles[1:])
    return [idx + 1 for idx in indices_b] if indices_b else None


def find_invalid_inds_between_lists(areas_a, areas_b, overlap_pairs):
    """Finds invalid indices based on area comparison between overlapping rectangles."""
    if not overlap_pairs: return [], []
    invalid_a, invalid_b = set(), set()
    for a_idx, b_idx in overlap_pairs:
        if areas_a[a_idx] < areas_b[b_idx]:
            invalid_a.add(a_idx)
        elif areas_b[b_idx] < areas_a[a_idx]:
            invalid_b.add(b_idx)
        else:
            invalid_a.add(a_idx)
            invalid_b.add(b_idx)
    return sorted(list(invalid_a)), sorted(list(invalid_b))


def find_invalid_inds(areas, overlap_indices):
    """Finds invalid indices based on area comparison with rectangle 0."""
    return sorted(list({idx for idx in overlap_indices if areas[idx] <= areas[0]})) if overlap_indices else []


def _find_transistivity(overlapping_indices):
    """Finds connected components in overlap groups."""
    g = nx.Graph()
    for node_set in overlapping_indices:
        g.add_edges_from(itertools.combinations(node_set, 2))
    return [tuple(sorted(c)) for c in nx.connected_components(g)]


def process_overlaps(rectangles_a, rectangles_b, x_region_max):
    """Processes overlaps to identify consecutive pairs based on row-major order."""
    _, _, overlap_pairs = find_overlapping_between_lists(rectangles_a, rectangles_b)
    if not overlap_pairs: return {}

    grouped_overlaps = defaultdict(list)
    for a_idx, b_idx in overlap_pairs:
        grouped_overlaps[a_idx].append(b_idx)

    result_dict = {}
    for a_idx, b_indices in grouped_overlaps.items():
        b_indices.sort()
        b_indices_set = set(b_indices)
        horizontal_pairs, vertical_pairs = [], []

        for b1 in b_indices:
            b2_h = b1 + 1
            if b2_h in b_indices_set and (b1 // x_region_max == b2_h // x_region_max):
                horizontal_pairs.append((b1, b2_h))
            b2_v = b1 + x_region_max
            if b2_v in b_indices_set:
                vertical_pairs.append((b1, b2_v))

        result_dict[a_idx] = (b_indices, len(b_indices), tuple(horizontal_pairs), tuple(vertical_pairs))
    return result_dict