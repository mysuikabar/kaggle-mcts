# https://www.kaggle.com/code/yunsuxiaozi/mcts-starter
import re


def ari(txt: str) -> float:
    characters = len(txt)
    words = len(re.split(" |\\n|\\.|\\?|\\!|\,", txt))
    sentence = len(re.split("\\.|\\?|\\!", txt))
    ari_score = 4.71 * (characters / words) + 0.5 * (words / sentence) - 21.43
    return ari_score


def mcalpine_eflaw(txt: str) -> float:
    W = len(re.split(" |\\n|\\.|\\?|\\!|\,", txt))
    S = len(re.split("\\.|\\?|\\!", txt))
    mcalpine_eflaw_score = (W + S * W) / S
    return mcalpine_eflaw_score


def clri(txt: str) -> float:
    characters = len(txt)
    words = len(re.split(" |\\n|\\.|\\?|\\!|\,", txt))
    sentence = len(re.split("\\.|\\?|\\!", txt))
    L = 100 * characters / words
    S = 100 * sentence / words
    clri_score = 0.0588 * L - 0.296 * S - 15.8
    return clri_score
