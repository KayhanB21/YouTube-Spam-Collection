from data_ingestion import read_dataset
import preprocessing
import helpers.pandas_h as hp
import spacy_word2vec as w2v
import utils
from config import settings
from logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("main function is stared.")

    utils.seed_everything(seed=settings.random_state)
    df = read_dataset()
    hp.null_val_summary(df=df)
    hp.unique_col_percent(df=df)

    df = preprocessing.deduplication(df=df)
    df = preprocessing.NLPPipeline(df=df).run()
    df = preprocessing.calc_author_spam_probability(df=df)

    word2vec = w2v.SpacyWord2Vec(df=df)
    word2vec.calc_avg_vector()
    word2vec.split_train_val_test()

    logger.info("main function is done.")


if __name__ == "__main__":
    main()
