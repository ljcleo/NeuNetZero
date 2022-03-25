from logging import Logger, StreamHandler, getLogger, FileHandler, Formatter
from pathlib import Path


def make_logger(name: str, root_path: Path, console: bool) -> Logger:
    logger: Logger = getLogger(name)
    logger.setLevel('INFO')
    formatter: Formatter = Formatter('[%(asctime)s %(name)s] %(levelname)s: %(message)s')

    if console:
        console_handler: StreamHandler = StreamHandler()
        console_handler.setLevel('INFO')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler: FileHandler = FileHandler(root_path / 'log' / f'{name}.log', 'w', encoding='utf8')
    file_handler.setLevel('INFO')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
