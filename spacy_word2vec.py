import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class SpacyWord2Vec:
    def __init__(self, df):
        self.avg_vect_tfidf_dataset = None
        self.avg_vect_dataset = None
        self.docs = None
        self._target = None
        self._avg_vect_tfidf = None
        self._avg_vect = None
        self._df = df

        try:
            self.nlp = spacy.load(settings.modeling.spacy.model, disable=['parser', 'ner'])
        except Exception as e:
            spacy.cli.download(settings.modeling.spacy.model)
            self.nlp = spacy.load(settings.modeling.spacy.model, disable=['parser', 'ner'])

    def calc_avg_vector(self):
        self._create_spacy_doc_list()
        self._get_avg_vector()
        self._get_avg_tfidf_vector()
        self._target = self._df['CLASS'].values

    def split_train_val_test(self):
        self.avg_vect_dataset = self._split_avg(avg_vect=self._avg_vect,
                                                target=self._target,
                                                name='avg w2c')
        self.avg_vect_tfidf_dataset = self._split_avg(avg_vect=self._avg_vect_tfidf,
                                                      target=self._target,
                                                      name='avg w2c with tf_idf')

    def _create_spacy_doc_list(self):
        self.docs = list(self.nlp.pipe(self._df['CONTENT_EDITED'],
                                       n_process=settings.modeling.spacy.number_of_process))

    def _get_avg_vector(self, ):
        """
        calculate the average vector of sentence using each word vector
        :return:
        :rtype:
        """
        avg_vect = []
        for doc in tqdm(self.docs):
            avg_vect.append(doc.vector)
        self._avg_vect = np.array(avg_vect)

    def _get_avg_tfidf_vector(self):
        """
        calculate the wighted average vector of sentence using each word vector and tf_idf method
        :return:
        :rtype:
        """
        tfidf_vectorizer = TfidfVectorizer(max_df=settings.modeling.tf_idf.max_df,
                                           min_df=settings.modeling.tf_idf.min_df)
        transformed_documents = tfidf_vectorizer.fit_transform(self._df['CONTENT_EDITED'])
        max_idf = max(tfidf_vectorizer.idf_)
        word2weight = defaultdict(lambda: max_idf,
                                  [(w, tfidf_vectorizer.idf_[i]) for w, i in tfidf_vectorizer.vocabulary_.items()])
        avg_vect_tfidf = []
        for doc in tqdm(self.docs):
            doc_vect = np.zeros(self.docs[0].vector.shape)
            for d in doc:
                doc_vect += word2weight[d] * d.vector
            doc_vect /= (len(doc) + 1)
            avg_vect_tfidf.append(doc_vect)
        self._avg_vect_tfidf = np.array(avg_vect_tfidf)

    @staticmethod
    def _split_avg(avg_vect: np.ndarray,
                   target: np.ndarray,
                   name: str) -> dict:
        x_train, x_test, y_train, y_test, inx_train, inx_test = train_test_split(avg_vect,
                                                                                 target,
                                                                                 [*range(len(avg_vect))],
                                                                                 test_size=settings.modeling.split.val +
                                                                                           settings.modeling.split.test)
        x_val, x_test, y_val, y_test, inx_val, inx_test = train_test_split(x_test,
                                                                           y_test,
                                                                           inx_test,
                                                                           test_size=settings.modeling.split.test / (
                                                                                   settings.modeling.split.val +
                                                                                   settings.modeling.split.test))
        logger.info(f"{name} info, "
                    f"{x_train.shape=}, "
                    f"{y_train.shape=}, "
                    f"{len(inx_train)=}, "
                    f"{x_val.shape=}, "
                    f"{y_val.shape=}, "
                    f"{len(inx_val)=}, "
                    f"{x_test.shape=}, "
                    f"{y_test.shape=}, "
                    f"{len(inx_test)=}")
        return {'x_train': x_train,
                'y_train': y_train,
                'inx_train': inx_train,
                'x_val': x_val,
                'y_val': y_val,
                'inx_val': inx_val,
                'x_test': x_test,
                'y_test': y_test,
                'inx_test': inx_test,
                }
