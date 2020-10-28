import os
import requests
import urllib
import logging
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)

API_KEY = os.environ.get("RAPIDAKEY")
if API_KEY is None:
    logger.warning("API KEY IS NOT SET, TRANSLATION WILL NOT WORK")


def translate(phrase: str,
              source_lang: str = "en",
              target_lang: str = "ru") -> str:
    """
    Universal function to translate from one language into another using
    rapidapi's approach to google translate API.
    """
    url = "https://google-translate1.p.rapidapi.com/language/translate/v2"
    payload = urllib.parse.urlencode({
        "q": phrase,
        "source": source_lang,
        "target": target_lang
    })
    headers = {
        'x-rapidapi-host': "google-translate1.p.rapidapi.com",
        'x-rapidapi-key': API_KEY,
        'accept-encoding': "application/gzip",
        'content-type': "application/x-www-form-urlencoded"
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    try:
        translation = response.json()["data"]["translations"][0]["translatedText"]
    except:
        translation = ""
        logger.exception(response.json())
    return translation


def translate_df(df: pd.DataFrame, columns: List[str], dump_file: Optional[str] = None):
    """
    Function to translate datafreames.
    :param df: - dataframe to translate
    :param columns: - columns to replace with translations
    :param dump_file: - optional filename to continuosly print lines to.
    Highly reccomended to use, because complete dataframe may be lost
    during the function
    :return:
    """

    splitter = "\t|\t"
    copy_df = df.copy()
    if dump_file is None and len(df) > 10:
        logger.warning("It is highly advised to specify `dump_file` for big dataframes.")
    if dump_file is not None:
        f = open(dump_file, "w")
        for c in columns:
            f.write(c)
            f.write(splitter)
        f.write("\n")
    try:
        for i in tqdm(range(len(df))):
            print(i)
            for k in columns:
                value = copy_df.iloc[i][k]
                translation = translate(value)
                copy_df.iloc[i][k] = translation
                if dump_file is not None:
                    f.write(translation)
                    f.write(splitter)
            if dump_file is not None:
                f.write("\n")
    except:
        logger.exception("Failed to translate")
    return copy_df
