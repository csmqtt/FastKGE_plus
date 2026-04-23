from .utils import *
from .model.model_process import *

class Trainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.train_processor = TrainBatchProcessor(args, kg)
        self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self) -> tuple[float, dict, dict]:
        self.args.valid = True
        loss, lora_stats = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res, lora_stats