from data_ingestion import read_dataset
import preprocessing
import helpers.pandas_h as hp
from config import settings
from logger import get_logger

logger = get_logger(__name__)


def main():
    pass


if __name__ == "__main__":
    logger.info("main function is stared.")
    df = read_dataset()
    hp.null_val_summary(df=df)
    hp.unique_col_percent(df=df)
    df = preprocessing.deduplication(df=df)
    pipe = preprocessing.NLPPipeline(df=df)
    pipe.run()
    logger.info("main function is done.")
