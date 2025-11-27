import torch
import numpy as np
import random
import logging
import sys
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_seeds(seed_num: int):

    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        
    logging.info(f"Initialized all random seeds to: {seed_num}")


def setup_logging(log_level: str = "INFO", log_file: str = "train.log") -> None:

    level = getattr(logging, log_level.upper(), logging.INFO)
    
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [] # (重要) 清除所有现有的 handlers

    # 2. 创建文件 Handler (写入到 'log_file')
    file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    # 3. 创建控制台 Handler (输出到 stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    logging.info("Logging setup complete. Outputting to console and file: %s", log_file)

def get_run_name(project_name: str) -> str:

    timestamp = datetime.now().strftime('%m%d_%H%M')
    return f"{timestamp}"