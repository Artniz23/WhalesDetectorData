class PipelineStep:
    """Базовый класс: любой модуль должен иметь метод run()."""
    def run(self, context):
        raise NotImplementedError