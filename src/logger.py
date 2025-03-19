import os
import logging
from datetime import datetime

class Logger:
    @staticmethod
    def setup_logger(model_name, log_dir="logs"):
        """
        로거 설정 함수
        """
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(f"{model_name}_training")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/{model_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger, log_file

    @staticmethod
    def log_metrics(logger, phase, fold=None, **metrics):
        if fold is not None:
            message = f"[{phase}] Fold {fold} - "
        else:
            message = f"[{phase}] Overall - "
        
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message += metrics_str
        
        logger.info(message)
