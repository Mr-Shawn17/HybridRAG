# log_config.py
import logging
import os
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler


class Log:
    @staticmethod
    def init_logger(
        log_path: str = "./logs/app.log",
        log_level=logging.INFO,
        file_level=None,
        console_level=None,
        when="D",
        interval=1,
        backupCount=7,  # 所有轮转文件总数（包含同一天内的多个分片），不是天数
        max_mb=10,  # 单个文件容量上限，单位MB
        use_gzip=False,
    ) -> logging.Logger:
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return root_logger

        root_logger.setLevel(log_level)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = ConcurrentTimedRotatingFileHandler(
            filename=log_path,
            when=when,
            interval=interval,
            backupCount=backupCount,
            maxBytes=max_mb * 1024 * 1024,
            encoding="utf-8",
            use_gzip=use_gzip,
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level if file_level is not None else log_level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        stream_handler.setLevel(
            console_level if console_level is not None else log_level
        )

        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        return root_logger
