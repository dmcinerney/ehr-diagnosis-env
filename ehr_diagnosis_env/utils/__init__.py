from operator import contains
from .model_interface import *
from .queries import *
import os
import pickle as pkl
from tqdm import tqdm


def sort_by_scores(items, scores):
    items_scores = sorted(zip(items, scores), key=lambda x: -x[1])
    return [x[0] for x in items_scores]


def update_cache_from_disk(cache, cache_dir):
    print("loading from cache: {}".format(cache_dir))
    files = [
        f for f in os.listdir(cache_dir)
        if f.startswith('cached_instance_') and f.endswith('.pkl')]
    pbar = tqdm(files, total=len(files))
    num_loaded, num_deleted = 0, 0
    for file in pbar:
        try:
            with open(os.path.join(cache_dir, file), 'rb') as f:
                cache.update(pkl.load(f))
            num_loaded += 1
        except Exception as e:
            # delete any files you can't load
            os.remove(os.path.join(cache_dir, file))
            num_deleted += 1
        pbar.set_postfix(
            {'num_loaded': num_loaded, 'num_deleted': num_deleted})


def map_confident_diagnosis(diagnosis, mapping_df):
    contains_rules = mapping_df[mapping_df.rule_type == 'contains']
    # these rules have the form that if a diagnosis contains x, then map
    # it to y
    matched_rules = contains_rules[
        contains_rules.x.apply(lambda x: x in diagnosis)]
    if len(matched_rules) == 0:
        return diagnosis
    elif len(matched_rules) == 1:
        return matched_rules.iloc[0].y
    else:
        raise Exception
