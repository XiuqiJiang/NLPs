import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    log_dir: str,
    name: str = "vae",
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        log_dir: 日志保存目录
        name: 日志记录器名称
        level: 日志级别
        console: 是否在控制台输出
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    
    # 创建控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    if console:
        console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    if console:
        logger.addHandler(console_handler)
    
    return logger 