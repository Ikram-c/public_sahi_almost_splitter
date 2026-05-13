import json
import argparse
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.utils.helper_functions import load_config
from pycocotools.coco import COCO


class CocoResolutionSplitter:

    def __init__(self, settings: Dict[str, Any]):
        self.input_path = Path(settings['input_json_path'])
        self.output_dir = Path(settings['output_dir'])

        self.json_indent = settings.get('json_indent', 4)
        self.filename_sep = settings.get('filename_separator', '_')
        self.res_sep = settings.get('resolution_separator', 'x')

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        print(f"Loading annotation file: {self.input_path}")
        self.coco = COCO(str(self.input_path))

        # This section robustly handles missing optional keys
        self.base_info = self.coco.dataset.get('info', {})
        self.base_licenses = self.coco.dataset.get('licenses', [])
        self.base_categories = self.coco.dataset.get('categories', [])

        self.name_part = self.input_path.stem
        self.ext_part = self.input_path.suffix

    def _group_images_by_resolution(self) -> defaultdict[tuple[int, int], list[int]]:
        print("Grouping images by resolution...")
        res_to_img_ids = defaultdict(list)
        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs([img_id])[0]
            resolution = (img['width'], img['height'])
            res_to_img_ids[resolution].append(img_id)

        print(f"Found {len(res_to_img_ids)} unique resolutions.")
        return res_to_img_ids

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        resolutions_map = self._group_images_by_resolution()

        # 1. Use a dictionary comprehension to create the path map in advance
        resolution_file_map = {
            (width,
             height): self.output_dir / f"{self.name_part}{self.filename_sep}{width}{self.res_sep}{height}{self.ext_part}"
            for (width, height) in resolutions_map.keys()
        }

        # 2. Loop through the original resolutions_map to do the processing
        for (width, height), img_ids in resolutions_map.items():
            res_str = f"{width}{self.res_sep}{height}"
            print(f"\nProcessing resolution: {res_str} ({len(img_ids)} images)")

            new_coco_data: Dict[str, Any] = {
                'info': self.base_info,
                'licenses': self.base_licenses,
                'categories': self.base_categories,
                'images': self.coco.loadImgs(img_ids),
            }

            ann_ids = self.coco.getAnnIds(imgIds=img_ids)
            new_coco_data['annotations'] = self.coco.loadAnns(ann_ids)

            # 3. Get the final path from the map you already created
            final_output_path = resolution_file_map[(width, height)]

            print(f"Saving to {final_output_path}...")
            with open(final_output_path, 'w') as f:
                json.dump(new_coco_data, f, indent=self.json_indent)

        print("\nSplitting complete.")

        # 4. Return the new dictionary
        return resolution_file_map

    @classmethod
    def run_from_config(cls, config_path: Path | str):
        print(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        splitter_settings = config.get('coco_splitter_settings')
        if not splitter_settings:
            raise KeyError(
                f"Key 'coco_splitter_settings' not found in {config_path}."
            )

        splitter = cls(splitter_settings)
        splitter.run()


def main():
    parser = argparse.ArgumentParser(
        description="Split a COCO JSON by resolution using a config file.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the main config.yaml file."
    )

    args = parser.parse_args()

    try:
        CocoResolutionSplitter.run_from_config(args.config_path)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()