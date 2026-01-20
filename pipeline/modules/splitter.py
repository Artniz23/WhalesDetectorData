import random
import shutil
from pathlib import Path
from typing import List, Tuple

from pipeline.context import PipelineContext
from pipeline.pipeline import PipelineStep


class DatasetSplitterModule(PipelineStep):
    def __init__(
            self,
            context: PipelineContext,
    ):
        self.source_root = Path(context.source_dir)
        self.output_root = Path(context.output_dir)

        assert abs(context.train_ratio + context.val_ratio + context.test_ratio - 1.0) < 1e-6

        self.train_ratio = context.train_ratio
        self.val_ratio = context.val_ratio
        self.test_ratio = context.test_ratio

        self.seed = context.seed
        random.seed(context.seed)

        self._prepare_output_dirs()

    def _prepare_output_dirs(self):
        for split in ["train", "val", "test"]:
            (self.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
            if split != "test":
                (self.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _split_folders(self):
        folders = [p for p in self.source_root.iterdir() if p.is_dir()]
        random.shuffle(folders)

        n = len(folders)

        if n < 2:
            raise ValueError("Need at least 2 folders to split dataset")

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        # Гарантии
        if n_val == 0 and n >= 2:
            n_val = 1

        if n_train + n_val >= n:
            n_train = n - n_val - 1

        train_folders = folders[:n_train]
        val_folders = folders[n_train:n_train + n_val]
        test_folders = folders[n_train + n_val:]

        return train_folders, val_folders, test_folders

    def _copy_folder(self, folder: Path, split: str):
        image_exts = {".jpg", ".jpeg"}

        # Рекурсивно проходить все файлы во всех вложенных папках
        for file in folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in image_exts:
                # Копируем изображение
                shutil.copy(
                    file,
                    self.output_root / "images" / split / file.name
                )

                # Копируем label (только для train и val)
                if split != "test":
                    label_file = file.with_suffix(".txt")
                    if label_file.exists():
                        shutil.copy(
                            label_file,
                            self.output_root / "labels" / split / label_file.name
                        )

    def run(self, context: PipelineContext):
        train_folders, val_folders, test_folders = self._split_folders()

        print(f"📦 Total whales: {len(train_folders) + len(val_folders) + len(test_folders)}")
        print(f"🟢 Train: {len(train_folders)} folders")
        print(f"🟡 Val:   {len(val_folders)} folders")
        print(f"🔵 Test:  {len(test_folders)} folders")

        for folder in train_folders:
            self._copy_folder(folder, "train")

        for folder in val_folders:
            self._copy_folder(folder, "val")

        for folder in test_folders:
            self._copy_folder(folder, "test")

        print("✅ Dataset split completed successfully")
