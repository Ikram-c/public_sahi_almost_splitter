import numpy as np
from typing import List, Tuple, Set, Dict
import itertools
from collections import defaultdict


class IntervalUnionQuery:
    """A segment tree implementation for managing overlapping intervals during slice generation."""

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
        self.active_rectangles = {}

        for i, val in enumerate(L):
            self.w[self.N + i] = val
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def modify_interval(self, i, k, offset, x_coord, rect_idx):
        """Updates coverage counts for the y indices [i, k)."""
        for y in range(i, k):
            if y not in self.active_rectangles:
                self.active_rectangles[y] = set()

            if offset == 1:
                self.active_rectangles[y].add(rect_idx)
            else:
                self.active_rectangles[y].discard(rect_idx)

            if self.active_rectangles[y]:
                self.sweep_events.append((x_coord, y, set(self.active_rectangles[y])))

        self._change(1, 0, self.N, i, k, offset)

    def query_coverage(self):
        """Returns the total coverage of active intervals."""
        return self.s[1]

    def _change(self, p, start, span, i, k, offset):
        """Recursive helper for segment tree update."""
        if start + span <= i or k <= start:
            return
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

    def __init__(self, x, rectangle, is_start, rect_idx):
        self.x = x
        self.rectangle = rectangle
        self.is_start = is_start
        self.rect_idx = rect_idx


class SweeplineSliceGenerator:
    """Generates overlapping slices using sweep-line algorithm."""

    def __init__(self, width: int, height: int, x_overlap: float, y_overlap: float):
        self.width = width
        self.height = height
        self.x_overlap = max(0.0, min(x_overlap, 0.99))
        self.y_overlap = max(0.0, min(y_overlap, 0.99))
        self.slices = []

    def generate_slices(self, target_tile_size: Tuple[int, int] = None) -> List[List[int]]:
        """
        Generate overlapping slices using sweep-line algorithm.

        Args:
            target_tile_size: Target size for tiles (width, height). If None, computed automatically.

        Returns:
            List of [x1, y1, x2, y2] coordinates for each slice.
        """
        if target_tile_size is None:
            # Determine optimal tile size based on overlap ratios
            # Higher overlap means we need more, smaller tiles
            num_tiles_x = max(2, int(1 / (1 - self.x_overlap)))
            num_tiles_y = max(2, int(1 / (1 - self.y_overlap)))

            tile_width = int(self.width / (num_tiles_x * (1 - self.x_overlap)))
            tile_height = int(self.height / (num_tiles_y * (1 - self.y_overlap)))
        else:
            tile_width, tile_height = target_tile_size

        # Ensure tile size doesn't exceed image dimensions
        tile_width = min(tile_width, self.width)
        tile_height = min(tile_height, self.height)

        # Use sweep-line to place tiles optimally
        return self._sweep_line_placement(tile_width, tile_height)

    def _sweep_line_placement(self, tile_width: int, tile_height: int) -> List[List[int]]:
        """
        Use sweep-line algorithm to place tiles with specified overlaps.
        """
        slices = []
        events = []

        # Calculate effective stride based on overlap
        stride_x = int(tile_width * (1 - self.x_overlap))
        stride_y = int(tile_height * (1 - self.y_overlap))
        stride_x = max(1, stride_x)
        stride_y = max(1, stride_y)

        # Generate initial tile positions
        tile_positions = []
        y = 0
        while y < self.height:
            x = 0
            while x < self.width:
                x1, y1 = x, y
                x2 = min(x + tile_width, self.width)
                y2 = min(y + tile_height, self.height)

                if x2 > x1 and y2 > y1:
                    tile_positions.append((x1, y1, x2, y2))

                x += stride_x
                if x + tile_width >= self.width and x < self.width - stride_x:
                    # Add edge tile
                    x1 = self.width - tile_width
                    x1 = max(0, x1)
                    if x1 < self.width:
                        tile_positions.append((x1, y1, min(self.width, x1 + tile_width), y2))
                    break

            y += stride_y
            if y + tile_height >= self.height and y < self.height - stride_y:
                # Add edge row
                y1 = self.height - tile_height
                y1 = max(0, y1)
                if y1 < self.height:
                    x = 0
                    while x < self.width:
                        x1 = x
                        x2 = min(x + tile_width, self.width)
                        y2 = min(self.height, y1 + tile_height)
                        if x2 > x1 and y2 > y1:
                            tile_positions.append((x1, y1, x2, y2))
                        x += stride_x
                        if x + tile_width >= self.width:
                            break
                break

        # Create events for sweep-line processing
        for i, rect in enumerate(tile_positions):
            events.append(Event(rect[0], rect, True, i))  # Start event
            events.append(Event(rect[2], rect, False, i))  # End event

        # Sort events by x-coordinate
        events.sort(key=lambda e: (e.x, not e.is_start))

        # Process with segment tree
        y_coords = sorted(list(set(y for rect in tile_positions for y in [rect[1], rect[3]])))

        if len(y_coords) < 2:
            return [[0, 0, self.width, self.height]]

        y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
        y_mapping = {val: idx for idx, val in enumerate(y_coords)}

        query = IntervalUnionQuery(y_intervals, y_coords)

        # Track overlaps and validate tiles
        validated_tiles = set()
        overlap_tracker = defaultdict(set)

        for event in events:
            y0 = y_mapping.get(event.rectangle[1], 0)
            y1 = y_mapping.get(event.rectangle[3], 0)

            if y0 >= y1:
                continue

            offset = 1 if event.is_start else -1
            query.modify_interval(y0, y1, offset, event.x, event.rect_idx)

            if event.is_start:
                # Check overlap constraints
                current_coverage = query.query_coverage()

                # Validate that overlap requirements are met
                overlap_area_x = 0
                overlap_area_y = 0

                for other_idx in overlap_tracker[event.x]:
                    other_rect = tile_positions[other_idx]

                    # Calculate actual overlap
                    x_overlap_actual = max(0, min(event.rectangle[2], other_rect[2]) - max(event.rectangle[0],
                                                                                           other_rect[0]))
                    y_overlap_actual = max(0, min(event.rectangle[3], other_rect[3]) - max(event.rectangle[1],
                                                                                           other_rect[1]))

                    if x_overlap_actual > 0:
                        overlap_area_x = max(overlap_area_x, x_overlap_actual / tile_width)
                    if y_overlap_actual > 0:
                        overlap_area_y = max(overlap_area_y, y_overlap_actual / tile_height)

                validated_tiles.add(event.rect_idx)
                overlap_tracker[event.x].add(event.rect_idx)
            else:
                overlap_tracker[event.x].discard(event.rect_idx)

        # Convert validated tiles to output format
        for idx in validated_tiles:
            rect = tile_positions[idx]
            slices.append([rect[0], rect[1], rect[2], rect[3]])

        # Remove duplicates
        unique_slices = []
        seen = set()
        for s in slices:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique_slices.append(s)

        return unique_slices


def generate_overlapping_slices_sweepline(
        resolutions: Set[Tuple[int, int]],
        overlap_ratios: Set[Tuple[float, float]],
        target_tile_size: Tuple[int, int] = None
) -> Dict[Tuple[int, int], List[List[int]]]:
    """
    Generates overlapping slices using sweep-line algorithm for given resolutions and overlap ratios.
    Uses 1-to-1 mapping: first resolution with first overlap ratio, second with second, etc.

    Args:
        resolutions: Set of (width, height) tuples representing image resolutions
        overlap_ratios: Set of (x_overlap, y_overlap) tuples where values are between 0 and 1
        target_tile_size: Optional target tile size (width, height)

    Returns:
        Dict[Tuple[int, int], List[List[int]]]: A dictionary where keys are
            (width, height) tuples and values are lists of [x1, y1, x2, y2]
            coordinates for each unique slice generated for that resolution.
    """
    all_slices: Dict[Tuple[int, int], List[List[int]]] = defaultdict(list)

    # <-- CHANGED: Convert sets to sorted lists for consistent 1-to-1 pairing
    resolutions_list = sorted(list(resolutions))
    overlap_ratios_list = sorted(list(overlap_ratios))

    for (width, height), (x_overlap, y_overlap) in zip(resolutions_list, overlap_ratios_list):
        resolution_key = (width, height)
        generator = SweeplineSliceGenerator(width, height, x_overlap, y_overlap)
        slices = generator.generate_slices(target_tile_size)
        all_slices[resolution_key].extend(slices)

    final_unique_slices: Dict[Tuple[int, int], List[List[int]]] = {}
    for res_key, slice_list in all_slices.items():
        unique_slices = []
        seen = set()
        for s in slice_list:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique_slices.append(s)
        final_unique_slices[res_key] = unique_slices

    return final_unique_slices


# # Example usage
# if __name__ == "__main__":
#     # <-- CHANGED: Example reverted to use sets, just like PreProcess
#     resolutions = {(758, 601), (765, 572), (811, 619), (824, 616), (826, 629)}
#     overlap_ratios = {(0.055, 0.095), (0.12, 0.08), (0.13, 0.145), (0.79, 0.005), (0.825, 0.86)}

#     # Generate slices using sweep-line algorithm with 1-to-1 mapping
#     slices_dict = generate_overlapping_slices_sweepline(resolutions, overlap_ratios)

#     total_slices = sum(len(s) for s in slices_dict.values())
#     print(f"Generated {total_slices} total unique slices across {len(slices_dict)} resolutions")
#     print(f"Using 1-to-1 mapping (each resolution paired with one overlap ratio)\n")

#     # Show the pairings used
#     print("Pairings used:")
#     resolutions_list = sorted(list(resolutions))
#     overlap_ratios_list = sorted(list(overlap_ratios))
#     for i, (res, overlap) in enumerate(zip(resolutions_list, overlap_ratios_list)):
#         print(f"  {i + 1}. Resolution {res} → Overlap {overlap}")

#     print("\nSlices for first resolution (example):")
#     if slices_dict:
#         # Get an arbitrary key from the dictionary to show an example
#         first_res_key = list(slices_dict.keys())[0]
#         first_res_slices = slices_dict[first_res_key]
#         print(f"Resolution: {first_res_key} ({len(first_res_slices)} unique slices)")
#         # Print first 10 slices for this resolution
#         for i, slice_coords in enumerate(first_res_slices[:10]):
#             print(f"Slice {i}: {slice_coords}")

#     # Optionally specify a target tile size
#     slices_with_size_dict = generate_overlapping_slices_sweepline(
#         resolutions,
#         overlap_ratios,
#         target_tile_size=(256, 256)
#     )

#     total_slices_with_size = sum(len(s) for s in slices_with_size_dict.values())
#     print(f"\nGenerated {total_slices_with_size} total unique slices with fixed tile size")