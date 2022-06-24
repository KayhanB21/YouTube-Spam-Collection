import random
import os
import numpy as np
import torch

from logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_training_device(force_gpu: bool = False):
    device_ids = 0
    if force_gpu:
        logger.info("force gpu is enabled.")
        device_ = torch.device("cuda")
        logger.info(f"{torch.cuda.device_count()=}")
        logger.info(f"{torch.cuda.current_device()=}")
        logger.info(f"{torch.cuda.get_device_name(torch.cuda.current_device())=}")
        logger.info(f"{torch.cuda.is_available()=}")
        device_ids = [i for i in range(torch.cuda.device_count())]
    else:
        logger.info("check whether gpu exists or not.")
        device_ = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"device is selected to be used for training: {device_}")
    return device_
