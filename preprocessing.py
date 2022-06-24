import pandas as pd
import re
from emot.emo_unicode import UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
import spacy
from tqdm import tqdm

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


class NLPPipeline:
    def __init__(self, df: pd.DataFrame):
        """
        Class that take care of content column and make sure the noise is being removed
        :param df:
        :type df:
        """
        self.df = df

        self.reg_flag_youtube = re.compile(
            r"youtu(?:.*\/v\/|.*v\=|\.be\/)([A-Za-z0-9_\-]{11})|watch\?v=([A-Za-z0-9_\-]{11})",
            re.IGNORECASE)

        self.reg_flag_url = re.compile(
            "(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$",
            re.IGNORECASE)

        self.emoticons = {k: v.replace(":", "").replace("_", " ").replace(",", "").strip() for k, v in
                          EMOTICONS_EMO.items()}
        self.unicode_emoji = {k: v.replace(":", "").replace("_", " ").replace(",", "").strip() for k, v in
                              UNICODE_EMOJI_ALIAS.items()}

    def run(self):
        """
        main routine for processing content column
        :return:
        :rtype:
        """
        logger.info("nlp processing is stared.")
        if settings.preprocessing.strip_text.enabled:
            self.strip_text()
        if settings.preprocessing.flag_youtube.enabled:
            self.flag_youtube()
        if settings.preprocessing.flag_url.enabled:
            self.flag_url()
        if settings.preprocessing.lower_case.enabled:
            self.lower_case()
        if settings.preprocessing.replace_emoji.enabled:
            self.replace_emoji()
        logger.info("nlp processing is finished.")

    def strip_text(self):
        """
        remove spaces from both side of the content
        :return: None
        :rtype: None
        """
        self.df['CONTENT'] = self.df['CONTENT'].str.strip()

    def flag_youtube(self):
        """
        to check whether the content has YouTube url
        :return: None
        :rtype: None
        """
        self.df['IS_YOUTUBE'] = self.df['CONTENT'].apply(lambda x: True if self.reg_flag_youtube.search(x) else False)

    def flag_url(self):
        """
        to check whether the content has an url or not
        :return: None
        :rtype: None
        """
        self.df['IS_YOUTUBE'] = self.df['CONTENT'].apply(lambda x: True if self.reg_flag_url.search(x) else False)

    def lower_case(self):
        """
        lower case all letters
        :return:
        :rtype:
        """
        self.df['CONTENT_EDITED'] = self.df['CONTENT'].apply(lambda x: x.casefold())

    def replace_emoji(self):
        """
        dataset has so many emojis which make sense to extract necessary information from comments.
        :return:
        :rtype:
        """

        def convert_emoji(text, debug=False):
            main_txt = text

            cnt1 = 0
            for emoj, emot_txt in self.emoticons.items():
                iter_txt = text
                pattern = '|'.join(
                    [' ' + re.escape(emoj) + ' ', '^(' + re.escape(emoj) + ')+', '(' + re.escape(emoj) + ')$'])
                text = re.sub(pattern, ' ' + emot_txt + ' ', text)
                if iter_txt != text:
                    cnt1 += 1

            cnt2 = 0
            for emoj, emot_txt in self.unicode_emoji.items():
                iter_txt = text
                text = re.sub(re.escape(emoj), ' ' + emot_txt + ' ', text)
                if iter_txt != text:
                    cnt2 += 1

            if main_txt != text and debug:
                print(main_txt, '\n', text)
                print(cnt1, '\t', cnt2)
                print('\n')
                print('\n')
            return text

        self.df['CONTENT_EDITED_TEMP'] = self.df['CONTENT_EDITED'].apply(lambda x: convert_emoji(x))
