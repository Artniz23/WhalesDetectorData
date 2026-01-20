class PipelineContext:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.seed: int = 42