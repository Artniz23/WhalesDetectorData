from pipeline.data_pipeline import DataPipeline

pipeline = (
    DataPipeline("dataset", "yolo_dataset")
    .withMaskPolygons()
    .withAugmentations()
    .withSplitter()
    .withVisualization()
)

pipeline.run()