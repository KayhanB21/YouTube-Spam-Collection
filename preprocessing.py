import pandas as pd

import helpers.pandas_h as hp
from config import settings
from logger import get_logger

logger = get_logger(__name__)


def deduplication(df: pd.DataFrame) -> pd.DataFrame:
    if len(settings.dataset.deduplication_columns) > 0:
        logger.info(f"executing deduplication for columns: {settings.dataset.deduplication_columns}")
        df = df.drop_duplicates(subset=settings.dataset.deduplication_columns)
        hp.unique_col_percent(df=df)
    return df
