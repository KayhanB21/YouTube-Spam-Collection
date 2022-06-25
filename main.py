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
    # data ingestion phase
    utils.seed_everything(seed=settings.random_state)
    df = read_dataset()
    hp.null_val_summary(df=df)
    hp.unique_col_percent(df=df)

    # data processing
    df = preprocessing.deduplication(df=df)
    df = preprocessing.NLPPipeline(df=df).run()
    df = preprocessing.calc_author_spam_probability(df=df)

    # conversion to vectors
    word2vec = w2v.SpacyWord2Vec(df=df)
    word2vec.calc_avg_vector()
    word2vec.split_train_val_test()

    # modeling phase and extract the best model in each approach
    model_perf = []
    model = ExtraTreesModel(avg_dict=word2vec.avg_vect_dataset,
                            avg_tf_idf_dict=word2vec.avg_vect_tfidf_dataset,
                            df=df,
                            spacy_docs=word2vec.docs)
    perf_etc_avg = model.find_best_hyperparameters_avg()
    model_perf.append((perf_etc_avg[0][0][2].mean(),
                       perf_etc_avg[1],
                       perf_etc_avg[2],
                       'Extra Trees Model Simple'))

    perf_etc_avg_tfidf = model.find_best_hyperparameters_avg_tfidf()
    model_perf.append((perf_etc_avg_tfidf[0][0][2].mean(),
                       perf_etc_avg_tfidf[1],
                       perf_etc_avg_tfidf[2],
                       'Extra Trees Model TF-IDF'))

    model = LogisticRegressionModel(avg_dict=word2vec.avg_vect_dataset,
                                    avg_tf_idf_dict=word2vec.avg_vect_tfidf_dataset,
                                    df=df,
                                    spacy_docs=word2vec.docs)

    perf_logistic_avg = model.find_best_hyperparameters_avg()
    model_perf.append((perf_logistic_avg[0][0][2].mean(),
                       perf_logistic_avg[1],
                       perf_logistic_avg[2],
                       'Logistic Regression Simple'))

    perf_logistic_avg_tfidf = model.find_best_hyperparameters_avg_tfidf()
    model_perf.append((perf_logistic_avg_tfidf[0][0][2].mean(),
                       perf_logistic_avg_tfidf[1],
                       perf_logistic_avg_tfidf[2],
                       'Logistic Regression TF-IDF'))

    # final report
    logger.info(f"summary of the extra trees model results "
                f"using simple average w2vec: \n{perf_etc_avg[0][1]}")
    logger.info(f"summary of the extra trees model results "
                f"using weighted tf-idf w2vec: \n{perf_etc_avg_tfidf[0][1]}")
    logger.info(f"summary of the logistic regression model results "
                f"using simple average w2vec: \n{perf_logistic_avg[0][1]}")
    logger.info(f"summary of the logistic regression results "
                f"using weighted tf-idf w2vec: \n{perf_logistic_avg_tfidf[0][1]}")

    # sort based on the models best average f1-score
    model_perf = sorted(model_perf, key=lambda tup: tup[0], reverse=True)

    logger.info(f"best model based on the average class recall is : {model_perf[0][3]}, with : {model_perf[0][2]}")
    logger.info("main function is done.")

    return model_perf[0]


if __name__ == "__main__":
    main()
