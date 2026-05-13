# TODO: Stream in using .Zarr converted dataset
# this is needed so we can stream the data for processing
from pycocotools.coco import COCO
from typing import Set, Tuple
from src.seg_splitter.image_slice_coords_finder import generate_overlapping_slices_sweepline

class PreProcess:
    def __init__(self, cfg: dict):
        self.tile_width = cfg.get("tile_width")
        self.tile_height = cfg.get("tile_height")
        self.unique_resolutions = self.get_unique_resolutions(cfg.get("ann_file_path"))
        self.overlap_ratios = self.get_overlap_ratios_per_resolution()
        self.resolution_to_slices = generate_overlapping_slices_sweepline(self.unique_resolutions,
                                                                          self.overlap_ratios)


    def calculate_overlap_ratios(self, img_size: tuple) -> tuple:
        """
        Compute overlap ratios needed for slicing.
        """

        x_res, y_res = img_size
        sw, sh = self.tile_width, self.tile_height
        x_mod, y_mod = x_res % sw, y_res % sh
        return (x_mod / sw if x_mod else 0.0,
                y_mod / sh if y_mod else 0.0)

    def get_unique_resolutions(self, ann_file_path: str) -> Set[Tuple[int, int]]:
        """Finds all unique image resolutions in a COCO annotation file.

        Args:
            ann_file_path: The file path to the COCO JSON annotation file.

        Returns:
            A set of tuples, where each tuple is a unique (width, height) pair.
        """
        coco = COCO(ann_file_path)
        unique_resolutions = set()
        for img_info in coco.imgs.values():
            unique_resolutions.add((img_info["width"], img_info["height"]))
        return unique_resolutions

    def get_overlap_ratios_per_resolution(self):

        overlap_ratios_set = set()
        for resolution in self.unique_resolutions:
            overlap_ratios = self.calculate_overlap_ratios(resolution)
            overlap_ratios_set.add(overlap_ratios)

        return overlap_ratios_set


