from pipeline.context import PipelineContext
from pipeline.modules.augmentation import AugmentationModule
from pipeline.modules.mask_to_polygon import MaskToPolygonModule
from pipeline.modules.splitter import DatasetSplitterModule
from pipeline.modules.visualize import PolygonVisualizerModule


class DataPipeline:
    def __init__(self, source_dir, output_dir):
        self.context = PipelineContext(source_dir, output_dir)
        self.steps = []

    # ====== BUILDER ======
    def withMaskPolygons(self):
        self.steps.append(MaskToPolygonModule(self.context))
        return self

    def withAugmentations(self, n_aug=3):
        self.steps.append(AugmentationModule(n_aug))
        return self

    def withSplitter(self):
        self.steps.append(DatasetSplitterModule(self.context))
        return self

    def withVisualization(self):
        self.steps.append(PolygonVisualizerModule(self.context))
        return self

    # ====== RUN PIPELINE ======
    def run(self):
        for step in self.steps:
            step.run(self.context)
        print("\nPipeline ✓ завершён!")
