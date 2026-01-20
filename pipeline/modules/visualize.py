import cv2
import numpy as np
from pathlib import Path
import os

from pipeline.context import PipelineContext
from pipeline.pipeline import PipelineStep


class PolygonVisualizerModule(PipelineStep):
    def __init__(
            self,
            context: PipelineContext
    ):
        self.dataset_root = Path(context.output_dir)
        self.output_root = Path(os.path.join(context.output_dir, "visualized"))
        self.image_exts = (".jpg", ".jpeg")
        self.color = (0, 255, 0)
        self.thickness = 2

        self._prepare_dirs()

    def _prepare_dirs(self):
        for split in ["train", "val"]:
            (self.output_root / split).mkdir(parents=True, exist_ok=True)

    def _load_polygon(self, label_path: Path, img_w: int, img_h: int):
        polygons = []

        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                coords = parts[1:]  # skip class id

                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_w)
                    y = int(coords[i + 1] * img_h)
                    points.append([x, y])

                polygons.append(np.array(points, dtype=np.int32))

        return polygons

    def _process_split(self, split: str):
        image_dir = self.dataset_root / "images" / split
        label_dir = self.dataset_root / "labels" / split
        output_dir = self.output_root / split

        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in self.image_exts:
                continue

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            polygons = self._load_polygon(label_path, w, h)

            for poly in polygons:
                cv2.polylines(
                    img,
                    [poly],
                    isClosed=True,
                    color=self.color,
                    thickness=self.thickness,
                )

            cv2.imwrite(str(output_dir / img_path.name), img)

    def run(self, context: PipelineContext):
        print("🎨 Visualizing polygons for train & val")

        for split in ["train", "val"]:
            self._process_split(split)

        print("✅ Polygon visualization completed")
