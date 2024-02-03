from .llm_queries.model_interface import get_model_interface
from .llm_queries.query import registered_queries
# you don't use this import, but it registers these queries
from .llm_queries.mistral_queries import *
from .llm_queries.flan_queries import *
from .llm_queries.alpacare_queries import *
import os
import pickle as pkl
from tqdm import tqdm
import warnings


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
    matched_targets = set(matched_rules.y)
    if len(matched_targets) == 0:
        return set([diagnosis])
    else:
        if len(matched_targets) > 1:
            warnings.warn(
                f'The diagnosis \"{diagnosis}\" matched to maps to mulitple '
                'targets: {}. Using all.'.format(', '.join(matched_targets)))
        return matched_targets


def confident_diagnosis_preprocessing(report_text):
    new_report_text = '\n'.join(
        [line for line in report_text.split('\n')
         if not line.lower().strip().startswith('admitting diagnosis: ')])
    return new_report_text
