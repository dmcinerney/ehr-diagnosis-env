from .model_interface import *
from .queries import *


def sort_by_scores(items, scores):
    items_scores = sorted(zip(items, scores), key=lambda x: -x[1])
    return [x[0] for x in items_scores]
