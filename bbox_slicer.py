from typing import Dict


class BboxSlicer:
    def __init__(self, config: Dict):
        self.config = config

    def get_slice_bboxes(
            self,
            image_height: int,
            image_width: int,
            slice_height: int | None = None,
            slice_width: int | None = None,
            overlap_height_ratio: float | None = 0.2,
            overlap_width_ratio: float | None = 0.2,
    ) -> list[list[int]]:
        """Generate bounding boxes for slicing an image into crops.

        The function calculates the coordinates for each slice based on the provided
        image dimensions, slice size, and overlap ratios. If slice size is not provided
        and auto_slice_resolution is True, the function will automatically determine
        appropriate slice parameters.

        from original sahi source -> with auto slice removed for now

        Args:
            image_height (int): Height of the original image.
            image_width (int): Width of the original image.
            slice_height (int, optional): Height of each slice. Default None.
            slice_width (int, optional): Width of each slice. Default None.
            overlap_height_ratio (float, optional): Fractional overlap in height of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.
            overlap_width_ratio(float, optional): Fractional overlap in width of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.

        Returns:
            List[List[int]]: List of 4 corner coordinates for each N slices.
                [
                    [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                    ...
                    [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
                ]
        """
        slice_bboxes = []
        y_max = y_min = 0

        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
        else:
            raise ValueError("Compute type is not auto and slice width and height are not provided.")

        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes
