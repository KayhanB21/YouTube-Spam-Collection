from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class ExtraTreesModel:
    def __init__(self, avg_dict: dict,
                 avg_tf_idf_dict: dict,
                 df,
                 spacy_docs):
        self.avg_dict = avg_dict
        self.avg_tf_idf_dict = avg_tf_idf_dict
        self.df = df
        self.spacy_docs = spacy_docs

    def find_best_hyperparameters_avg(self):
        best_estimator, best_params = self.hyperparameter_search(dataset_dict=self.avg_dict)
        return self.test_best_model(best_estimator=best_estimator,
                                    dataset_dict=self.avg_dict,
                                    confusion_matrix_filename=f'extra_trees_classifier_avg_vector'), best_estimator, best_params

    def find_best_hyperparameters_avg_tfidf(self):
        best_estimator, best_params = self.hyperparameter_search(dataset_dict=self.avg_tf_idf_dict)
        return self.test_best_model(best_estimator=best_estimator,
                                    dataset_dict=self.avg_tf_idf_dict,
                                    confusion_matrix_filename=f'extra_trees_classifier_avg_vector_tfidf'), best_estimator, best_params

    @staticmethod
    def hyperparameter_search(dataset_dict: dict):
        clf = ExtraTreesClassifier()
        param_grid = {"n_estimators": settings.modeling.extra_trees_classifiers.n_estimators}
        ps = PredefinedSplit(np.append(np.full((len(dataset_dict['x_train']),), -1, dtype=int),
                                       np.full((len(dataset_dict['x_val']),), 0, dtype=int)))
        grid_search = GridSearchCV(clf, param_grid, cv=ps, verbose=2, n_jobs=-1)
        grid_search.fit(np.concatenate((dataset_dict['x_train'], dataset_dict['x_val'])),
                        np.concatenate((dataset_dict['y_train'], dataset_dict['y_val'])))
        logger.info(f"The best hyperparameters are {grid_search.best_params_}, "
                    f"with score: {grid_search.best_score_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def test_best_model(self, best_estimator,
                        dataset_dict: dict,
                        confusion_matrix_filename: str):
        y_pred = best_estimator.predict(dataset_dict['x_test'])
        incorrect_inx = np.where((y_pred == dataset_dict['y_test']) == False)[0]
        max_count_example = settings.modeling.extra_trees_classifiers.count_demo_false_prediction if len(
            incorrect_inx) > settings.modeling.extra_trees_classifiers.count_demo_false_prediction else len(
            incorrect_inx)
        for i in range(max_count_example):
            selected_inx = dataset_dict['inx_test'][incorrect_inx[i]]
            logger.info(f"doc: {self.df['CONTENT'].iloc[selected_inx]}, \n"
                        f"edited doc: {self.spacy_docs[selected_inx]}")
            logger.info(f"predicted class: {y_pred[incorrect_inx[i]]}, \n"
                        f"dataset class: {dataset_dict['y_test'][incorrect_inx[i]]}, \n"
                        f"original class: {self.df['CLASS'].iloc[selected_inx]}")
        cm = confusion_matrix(dataset_dict['y_test'], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=best_estimator.classes_)
        disp.plot()
        plt.savefig(f'plots/{confusion_matrix_filename}.jpg')
        return precision_recall_fscore_support(dataset_dict['y_test'], y_pred), classification_report(
            dataset_dict['y_test'], y_pred, labels=best_estimator.classes_)
