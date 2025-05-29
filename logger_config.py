import logging
import os
from datetime import datetime

def setup_logger(cur_time=None):
    """设置同时输出到文件和终端的logger"""
    if cur_time is None:
        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # 创建logger
    logger = logging.getLogger("rag_system")
    logger.setLevel(logging.INFO)

    # 如果logger已经有handler，则清除（避免重复添加）
    if logger.handlers:
        logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建文件handler
    log_filename = f"logs/rag_system_{cur_time}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)  # 确保日志目录存在
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建终端handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
