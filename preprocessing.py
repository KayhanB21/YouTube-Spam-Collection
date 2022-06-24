import pandas as pd
import re
from emot.emo_unicode import UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
import spacy
from tqdm import tqdm
from pandarallel import pandarallel
import psutil

import helpers.pandas_h as hp
from config import settings
from logger import get_logger

logger = get_logger(__name__)
if not settings.preprocessing.replace_emoji.number_of_process:
    settings.preprocessing.replace_emoji.number_of_process = psutil.cpu_count(logical=False)
pandarallel.initialize(progress_bar=True, nb_workers=settings.preprocessing.replace_emoji.number_of_process)


class NLPPipeline:
    def __init__(self, df: pd.DataFrame):
        """
        Class that take care of content column and make sure the noise is being removed
        :param df: main dataframe from reading csv
        :type df:
        """
        self.df = df
        self.df['CONTENT_EDITED'] = self.df['CONTENT']

        # find YouTube link
        self.reg_flag_youtube = re.compile(
            r"youtu(?:.*\/v\/|.*v\=|\.be\/)([A-Za-z0-9_\-]{11})|watch\?v=([A-Za-z0-9_\-]{11})",
            re.IGNORECASE)

        # find url
        self.reg_flag_url = re.compile(
            "(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$",
            re.IGNORECASE)

        # emoji
        emoticons = {k: v.replace(":", "").replace("_", " ").replace(",", "").strip() for k, v in
                     EMOTICONS_EMO.items()}
        unicode_emoji = {k: v.replace(":", "").replace("_", " ").replace(",", "").strip() for k, v in
                         UNICODE_EMOJI_ALIAS.items()}
        self.emoticons = []
        for emoj, emot_txt in emoticons.items():
            self.emoticons.append(('|'.join([' ' + re.escape(emoj) + ' ',
                                             '^(' + re.escape(emoj) + ')+',
                                             '(' + re.escape(emoj) + ')$']), emot_txt))
        self.unicode_emoji = []
        for emoj, emot_txt in unicode_emoji.items():
            self.unicode_emoji.append((re.escape(emoj), emot_txt))

        # lemmatization
        try:
            self.nlp = spacy.load(settings.preprocessing.lemmatization.spacy_model, disable=['parser', 'ner'])
        except Exception as e:
            spacy.cli.download(settings.preprocessing.lemmatization.spacy_model)
            self.nlp = spacy.load(settings.preprocessing.lemmatization.spacy_model, disable=['parser', 'ner'])

        logger.info(f"loaded spacy pipes: {self.nlp.pipe_names}")

    def run(self) -> pd.DataFrame:
        """
        main routine for processing content column
        :return:
        :rtype:
        """
        logger.info("word processing is stared.")
        if settings.preprocessing.strip_text.enabled:
            logger.info("strip_text is stared.")
            self.strip_text()
        if settings.preprocessing.flag_youtube.enabled:
            logger.info("flag_youtube is stared.")
            self.flag_youtube()
        if settings.preprocessing.flag_url.enabled:
            logger.info("flag_url is stared.")
            self.flag_url()
        if settings.preprocessing.lower_case.enabled:
            logger.info("lower_case is stared.")
            self.lower_case()
        if settings.preprocessing.replace_emoji.enabled:
            logger.info("replace_emoji is stared.")
            self.replace_emoji()
        if settings.preprocessing.remove_special_characters.enabled:
            logger.info("remove_special_characters is stared.")
            self.remove_special_characters()
        if settings.preprocessing.remove_numbers.enabled:
            logger.info("remove_numbers is stared.")
            self.remove_numbers()
        if settings.preprocessing.remove_short_word.enabled:
            logger.info("remove_short_word is stared.")
            self.remove_short_word()
        if settings.preprocessing.remove_extra_white_space.enabled:
            logger.info("remove_extra_white_space is stared.")
            self.remove_extra_white_space()
        if settings.preprocessing.lemmatization.enabled:
            logger.info("lemmatization is stared.")
            self.lemmatization()

        logger.info("word processing is finished.")
        return self.df

    def strip_text(self):
        """
        remove spaces from both side of the content
        :return: None
        :rtype: None
        """
        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].str.strip()

    def flag_youtube(self):
        """
        to check whether the content has YouTube url
        :return: None
        :rtype: None
        """
        self.df['IS_YOUTUBE'] = self.df['CONTENT_EDITED'].apply(
            lambda x: True if self.reg_flag_youtube.search(x) else False)

    def flag_url(self):
        """
        to check whether the content has an url or not
        :return: None
        :rtype: None
        """
        self.df['IS_URL'] = self.df['CONTENT_EDITED'].apply(
            lambda x: True if self.reg_flag_url.search(x) else False)

    def lower_case(self):
        """
        lower case all letters
        # Before: THIS TEXT WILL BE LOWERCASED. THIS too: ßßß
        # After: this text will be lowercased. this too: ssssss
        # source: https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/#tokenize-text-using-spacy
        :return: None
        :rtype: None
        """
        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].apply(lambda x: x.casefold())

    def replace_emoji(self):
        """
        dataset has so many emojis which make sense to extract necessary information from comments.
        :return: None
        :rtype: None
        """

        def convert_emoji(text):
            main_txt = text

            cnt1 = 0
            for pattern, emot_txt in self.emoticons:
                iter_txt = text
                text = re.sub(pattern, ' ' + emot_txt + ' ', text)
                if iter_txt != text:
                    cnt1 += 1

            cnt2 = 0
            for pattern, emot_txt in self.unicode_emoji:
                iter_txt = text
                text = re.sub(pattern, ' ' + emot_txt + ' ', text)
                if iter_txt != text:
                    cnt2 += 1

            if main_txt != text:
                logger.debug(f"{main_txt}, '\n'{text}")
                logger.debug(f"{cnt1}, {cnt2}")
            return text

        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].parallel_apply(convert_emoji, )

    def remove_special_characters(self):
        """
        remove special characters from string
        # Before: Sample text 123 !!!! Haha.... !!!! ##$$$%%%%
        # After: Sample text 123  Haha

        :return: None
        :rtype: None
        """

        def exec_func(text):
            # return re.sub(r"\b[0-9]+\b\s*", " ", text)
            return re.sub(r"[^A-Za-z0-9\s]+", " ", text)

        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].apply(lambda x: exec_func(x))

    def remove_numbers(self):
        """
        remove number from string
        # Before: Remove these numbers: 1919191 2229292 11.233 22/22/22.
        # After: Remove these numbers: .//.
        :return: None
        :rtype: None
        """

        def exec_func(text):
            # return re.sub(r"\b[0-9]+\b\s*", " ", text)
            return ''.join([i for i in text if not i.isdigit()])

        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].apply(lambda x: exec_func(x))

    def remove_short_word(self):
        """
        removing word with two character and less
        :return: None
        :rtype: None
        """

        def exec_func(text):
            return re.sub(r'\b\w{1,2}\b', '', text)

        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].apply(lambda x: exec_func(x))

    def remove_extra_white_space(self):
        """
        extra white spaces can be removed and be reduced to one space
        :return: None
        :rtype: None
        """

        def exec_func(text):
            return re.sub(r'^\s*|\s\s*', ' ', text).strip()

        self.df['CONTENT_EDITED'] = self.df['CONTENT_EDITED'].apply(lambda x: exec_func(x))

    def lemmatization(self):
        """
        finding each word root in the language and make sure words has pretrained embedding vectors.
        :return: None
        :rtype: None
        """
        unique_desc = list(self.df['CONTENT_EDITED'].unique())
        res = []
        for doc in tqdm(self.nlp.pipe(unique_desc, n_process=settings.preprocessing.lemmatization.number_of_process)):
            res.append(" ".join(token.lemma_ for token in doc if (not token.is_stop) and token.has_vector))
        for inx, item in enumerate(tqdm(unique_desc)):
            self.df.loc[self.df[self.df['CONTENT_EDITED'] == item].index, 'CONTENT_EDITED'] = res[inx]


def deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove duplicated rows (as it was discovered in exploration phase)
    :param df:
    :type df:
    :return:
    :rtype:
    """
    if len(settings.preprocessing.deduplication_columns) > 0:
        logger.info(f"executing deduplication for columns: {settings.preprocessing.deduplication_columns}")
        df = df.drop_duplicates(subset=settings.preprocessing.deduplication_columns)
        hp.unique_col_percent(df=df)
    return df


def calc_author_spam_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculate the probability of author being spam based on the historic data
    :param df:
    :type df:
    :return:
    :rtype:
    """
    res = df[df['CLASS'] == 1].groupby('AUTHOR').size() / df.groupby('AUTHOR').size()
    res = res.fillna(0)
    for inx, item in tqdm(res.iteritems()):
        df.loc[df[df['AUTHOR'] == inx].index, 'AUTHOR_SPAM_PROB'] = item
    return df
