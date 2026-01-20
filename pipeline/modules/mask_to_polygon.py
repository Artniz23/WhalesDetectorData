import cv2
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.pipeline import PipelineStep


class MaskToPolygonModule(PipelineStep):
    def __init__(self, context: PipelineContext):
        self.source_root = Path(context.source_dir)
        self.class_id = 0

    def _mask_to_polygon(self, mask_path: Path, img_w: int, img_h: int):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for cnt in contours:
            if len(cnt) < 3:
                continue

            cnt = cnt.squeeze()
            poly = []
            for x, y in cnt:
                poly.append(x / img_w)
                poly.append(y / img_h)

            polygons.append(poly)

        return polygons

    def run(self, context: PipelineContext):
        for folder in self.source_root.iterdir():
            if not folder.is_dir():
                continue

            for img_path in folder.rglob("*.jpg"):
                mask_path = img_path.with_suffix(".png")
                if not mask_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]

                polygons = self._mask_to_polygon(mask_path, w, h)
                if not polygons:
                    continue

                txt_path = img_path.with_suffix(".txt")
                with open(txt_path, "w") as f:
                    for poly in polygons:
                        line = " ".join(
                            [str(self.class_id)] +
                            [f"{v:.6f}" for v in poly]
                        )
                        f.write(line + "\n")