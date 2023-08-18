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
            # print(e)
            import pdb; pdb.set_trace()
            # delete any files you can't load
            os.remove(os.path.join(cache_dir, file))
            num_deleted += 1
        pbar.set_postfix(
            {'num_loaded': num_loaded, 'num_deleted': num_deleted})
