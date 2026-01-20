import albumentations as A
import cv2
import os

from pipeline.context import PipelineContext
from pipeline.pipeline import PipelineStep

class AugmentationModule(PipelineStep):
    def __init__(self, n_aug=3):
        self.n_aug = n_aug
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.2, p=0.7),
            A.HueSaturationValue(10, 30, 10, p=0.7)
        ], additional_targets={"mask": "mask"})

    def run(self, context: PipelineContext):
        input_dir = context.source_dir
        output_dir = os.path.join(context.output_dir, "augmented")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Start Augmentation input dir: {input_dir} and output dir: {output_dir}")

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".jpg"):
                    continue

                img_path = os.path.join(root, file)
                mask_path = img_path.replace(".jpg", ".png")

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print("Cannot read image:", img_path)
                    continue
                if mask is None:
                    print("Cannot read mask:", mask_path)
                    continue

                # Поддерживаем структуру папок
                rel = os.path.relpath(root, input_dir)
                out_dir_current = os.path.join(output_dir, rel)
                os.makedirs(out_dir_current, exist_ok=True)

                # save original
                cv2.imwrite(os.path.join(out_dir_current, file), img)
                cv2.imwrite(os.path.join(out_dir_current, file.replace(".jpg", ".png")), mask)

                # augment
                for i in range(self.n_aug):
                    aug = self.transform(image=img, mask=mask)
                    aug_img = aug["image"]
                    aug_mask = aug["mask"]

                    name = file.replace(".jpg", f"_aug{i}")
                    cv2.imwrite(os.path.join(out_dir_current, name + ".jpg"), aug_img)
                    cv2.imwrite(os.path.join(out_dir_current, name + ".png"), aug_mask)

        # обновляем каталог текущих изображений
        context.source_dir = output_dir
