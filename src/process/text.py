import re
import string

from .consts import STOP_WORDS


def preprocess_text(text: str) -> str:
    text = text.lower()

    # replace punctuation with space
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])

    return text
