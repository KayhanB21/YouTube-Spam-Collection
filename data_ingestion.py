import pandas as pd

from config import settings
from logger import get_logger

logger = get_logger(__name__)


def read_dataset() -> pd.DataFrame:
    """
    read files based on the provided config
    :return:
    :rtype:
    """
    df = pd.DataFrame()
    for file_name in settings.dataset.csv_files:
        logger.info(f"start reading {file_name}")
        df_temp = pd.read_csv(f'dataset/{file_name}', engine='python', encoding='utf-8-sig')
        df_temp['video_id'] = file_name
        df = pd.concat([df, df_temp])
        logger.info(f"reading {file_name} is finished, "
                    f"number of new records: {len(df_temp.index)}, "
                    f"number of total records: {len(df.index)}")
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace("\ufeff", ""))
    df = df.reset_index(drop=True)
    return df
