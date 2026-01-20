from pipeline.data_pipeline import DataPipeline

pipeline = (
    DataPipeline("dataset", "yolo_dataset")
    .WithMaskPolygons()
    .withSplitter()
)

pipeline.run()