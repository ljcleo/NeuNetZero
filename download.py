from gzip import decompress
from hashlib import new
from logging import Logger
from pathlib import Path

from requests import Response, get

from src.tool.logger import make_logger
from src.tool.util import get_path


def download(url: str, md5: str, path: Path) -> bool:
    if path.exists():
        try:
            with path.open('rb') as f:
                data: bytes = f.read()
            if new('md5', data).hexdigest() == md5:
                return True
        except Exception:
            pass

    try:
        response: Response = get(url)
        if response.status_code != 200:
            return False

        data: bytes = decompress(response.content)
        if new('md5', data).hexdigest() != md5:
            return False

        with path.open('wb') as f:
            f.write(data)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    root_path: Path = Path('.')
    training_path: Path = get_path(root_path, 'data', 'training')
    test_path: Path = get_path(root_path, 'data', 'test')
    mnist_url: str = 'http://yann.lecun.com/exdb/mnist'

    logger: Logger = make_logger('download', root_path, True, direct=True)
    logger.info('Start downloading MNIST dataset ...')

    for url, md5, path in [
        (f'{mnist_url}/train-images-idx3-ubyte.gz', '6bbc9ace898e44ae57da46a324031adb',
         training_path / 'images.bin'),
        (f'{mnist_url}/train-labels-idx1-ubyte.gz', 'a25bea736e30d166cdddb491f175f624',
         training_path / 'labels.bin'),
        (f'{mnist_url}/t10k-images-idx3-ubyte.gz', '2646ac647ad5339dbf082846283269ea',
         test_path / 'images.bin'),
        (f'{mnist_url}/t10k-labels-idx1-ubyte.gz', '27ae3e4e09519cfbb04c329615203637',
         test_path / 'labels.bin'),
    ]:
        for i in range(3):
            logger.info(f'{"Retry downloading" if i > 0 else "Downloading"} from {url}')

            if download(url, md5, path):
                logger.info(f'Finished writing data to {path}.')
                break
        else:
            logger.error(f'Failed to download from {url} to {path}!')
            raise RuntimeError(f'cannot download from {url} to {path}')

    logger.info('Finished downloading MNIST dataset.')
