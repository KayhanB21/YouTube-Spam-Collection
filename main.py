from data_ingestion import read_dataset
import preprocessing
import helpers.pandas_h as hp
import spacy_word2vec as w2v
from extra_trees_model import ExtraTreesModel
from logistic_regression_model import LogisticRegressionModel
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

    model = ExtraTreesModel(avg_dict=word2vec.avg_vect_dataset,
                            avg_tf_idf_dict=word2vec.avg_vect_tfidf_dataset,
                            df=df,
                            spacy_docs=word2vec.docs)
    perf_etc_avg = model.find_best_hyperparameters_avg()
    perf_etc_avg_tfidf = model.find_best_hyperparameters_avg_tfidf()

    model = LogisticRegressionModel(avg_dict=word2vec.avg_vect_dataset,
                                    avg_tf_idf_dict=word2vec.avg_vect_tfidf_dataset,
                                    df=df,
                                    spacy_docs=word2vec.docs)
    perf_logistic_avg = model.find_best_hyperparameters_avg()
    perf_logistic_avg_tfidf = model.find_best_hyperparameters_avg_tfidf()

    logger.info(f"summary of the extra trees model results using simple average w2vec: \n{perf_etc_avg[1]}")
    logger.info(f"summary of the extra trees model results using weighted tf-idf w2vec: \n{perf_etc_avg_tfidf[1]}")
    logger.info(f"summary of the logistic regression model results using simple average w2vec: \n{perf_logistic_avg[1]}")
    logger.info(f"summary of the logistic regression results using weighted tf-idf w2vec: \n{perf_logistic_avg_tfidf[1]}")

    logger.info("main function is done.")


if __name__ == "__main__":
    main()
