import os

import gymnasium as gym
import torch
from gymnasium import spaces
from ehr_diagnosis_env.utils import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import io
import string
from sentence_transformers import SentenceTransformer, util
from .. import utils
from importlib.resources import files
import pickle as pkl


# TODO: add in a more specific manual list of alternative diagnoses
# TODO: remove all '.' when reading the alternative diagnoses csv


class EHRDiagnosisEnv(gym.Env):
    def __init__(self, instances=None, top_k_evidence=3, model_name='google/flan-t5-xxl', fuzzy_matching_threshold=.85,
                 progress_bar=None, reward_type='continuous_independent', include_alternative_diagnoses=True,
                 num_future_diagnoses_threshold=1, match_confident_diagnoses=True, true_positive_minimum=1,
                 cache_path=None, limit_targets_to=None, verbosity=2):
        """
        :param instances: A dataframe of patient instances with one column of called 'reports' where each element is a
            dataframe of reports ordered by date. The dataframes are in string csv format with one column called 'text'.
        :param top_k_evidence: an int determining how many diagnoses for which to query evidence
        """
        self._all_instances = None
        self._extracted_information_cache = None
        self.cache_path = None
        if instances is not None:
            self.set_instances(instances, cache_path=cache_path)
        self._seen_instances = set()
        self.top_k_evidence = top_k_evidence
        self.model_name = model_name
        self.fuzzy_matching_threshold = fuzzy_matching_threshold
        self.progress_bar = progress_bar if progress_bar is not None else tqdm
        self.reward_type = reward_type
        assert self.reward_type in ['continuous_independent', 'continuous_dependent', 'ranking']
        self.include_alternative_diagnoses = include_alternative_diagnoses
        self.num_future_diagnoses_threshold = num_future_diagnoses_threshold
        self.match_confident_diagnoses = match_confident_diagnoses
        self.true_positive_minimum = true_positive_minimum
        self.cache_path = cache_path
        self.limit_targets_to = None
        self.set_limit_targets_to(limit_targets_to)
        self.verbosity = verbosity

        self.seed = None
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'))
        self.observation_space = spaces.Dict(
            {
                # reports seen up to the current timestep in csv format
                "reports": spaces.Text(10000, charset=string.printable),
                # potential diagnoses seen up to the current timestep in csv format
                "potential_diagnoses": spaces.Text(10000, charset=string.printable),
                # evidence extracted from previous reports for each diagnosis in csv format
                "evidence": spaces.Text(10000, charset=string.printable),
                # whether the evidence for this report was retrieved (determines what kind of action you are taking)
                "evidence_is_retrieved": spaces.Discrete(2),
            }
        )

        # denotes the index of the currently observed note, leave to the reset method to set this
        self._current_report_index = None

        # denotes the report index at which observations start
        # (reports before this index are never directly shown to the agent but are used during evidence retrieval)
        self._start_report_index = None

        # denotes if the action for evidence retrieval has been taken for the currently observed note,
        # leave to the reset method to set this
        self._evidence_is_retrieved = None

        # a running dictionary of dictionaries keeping track of retrieved evidence for each diagnosis for each report
        self._current_evidence = None

        self._all_reports = None
        self._extracted_information = None

        print('loading model interface')
        self.model = get_model_interface(model_name)
        self.fuzzy_matching_model = None
        if self.fuzzy_matching_threshold is not None:
            self.fuzzy_matching_model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.include_alternative_diagnoses or self.match_confident_diagnoses:
            alternatives1 = pd.read_csv(files(utils) / 'gpt_alternative_diagnoses.txt', delimiter='\t')
            alternatives2 = pd.read_csv(files(utils) / 'manual_alternatives.txt', delimiter='\t')
            self.alternatives = pd.concat([alternatives1, alternatives2])
            self.alternatives['full_set'] = self.alternatives.apply(
                lambda r: set([r.diagnosis] + [a.strip() for a in r.alternatives.split(',')]), axis=1)
        if self.match_confident_diagnoses:
            self.all_reference_diagnoses = set().union(*self.alternatives.full_set.to_list())

    def to(self, device):
        self.model.to(device)
        self.fuzzy_matching_model.to(device)

    def set_instances(self, instances, cache_path=None):
        self._all_instances = instances
        self._extracted_information_cache = {}
        self.cache_path = cache_path
        if self.cache_path is not None:
            if not os.path.exists(self.cache_path):
                os.mkdir(self.cache_path)
            for file in os.listdir(self.cache_path):
                if file.startswith('cached_instance_'):
                    with open(os.path.join(self.cache_path, file), 'rb') as f:
                        self._extracted_information_cache.update(pkl.load(f))

    def set_limit_targets_to(self, limit_targets_to):
        self.limit_targets_to = limit_targets_to

    def extract_info(self, reports):
        extracted_information = {
            'differential diagnoses': [],
            'risk prediction': [],
            'confident diagnoses': [],
            'potential diagnoses': [],
            'target diagnoses': [],
            'true positives': [],
            'past target diagnoses': [],
        }
        diagnosis_target_key = 'matched confident diagnoses' \
            if self.match_confident_diagnoses else 'confident diagnoses'
        if self.match_confident_diagnoses:
            extracted_information['matched confident diagnoses'] = []
        for i, row in self.progress_bar(
                reports.iterrows(), total=len(reports), desc='extracting information from reports'):
            first_queries = ['differential diagnoses', 'risk prediction', 'confident diagnosis exists']
            out = self.model.query(
                tuple(row.text for _ in range(len(first_queries))),
                tuple(queries[query_name] for query_name in first_queries))
            extracted_information['differential diagnoses'].append(process_set_output(out['output'][0]))
            extracted_information['risk prediction'].append(process_set_output(out['output'][1]))
            confident_diagnosis_exists = process_string_output(out['output'][2])
            if confident_diagnosis_exists == 'yes':
                out = self.model.query((row.text,), (queries['confident diagnosis'],))
                out = self.model.query(
                    ('',),
                    (queries['confident diagnoses extracted'].replace('<confident diagnosis>', out['output'][0]),))
                extracted_information['confident diagnoses'].append(process_set_output(out['output'][0]))
            else:
                extracted_information['confident diagnoses'].append(set())
            if self.match_confident_diagnoses:
                extracted_information['matched confident diagnoses'].append(
                    self.get_matched_diagnoses(
                        extracted_information['confident diagnoses'][-1], self.all_reference_diagnoses))
            # keep a running list of confident diagnoses seen up to and including the current time-points
            previous_past_targets = set() if len(extracted_information['past target diagnoses']) == 0 else \
                extracted_information['past target diagnoses'][-1]
            extracted_information['past target diagnoses'].append(
                previous_past_targets.union(extracted_information[diagnosis_target_key][-1]))
        # initially set to the union of all confident diagnoses that emerge in the episode
        for differential_diagnoses, risks, target_diagnoses, past_targets in zip(
                extracted_information['differential diagnoses'],
                extracted_information['risk prediction'],
                extracted_information[diagnosis_target_key],
                extracted_information['past target diagnoses'],
        ):
            # eliminate the targets that appear in the report from the list of targets for that and future time-point
            previous_targets = extracted_information['past target diagnoses'][-1] \
                if len(extracted_information['target diagnoses']) == 0 else \
                extracted_information['target diagnoses'][-1]
            extracted_information['target diagnoses'].append(previous_targets.difference(target_diagnoses))
            # keep a running list of potential diagnoses by combining differentials and risks
            new_potential_diagnoses = differential_diagnoses.union(risks)
            if self.include_alternative_diagnoses:
                # if including alternatives, expand the list by adding anything that appears in the same line (in the
                # alternatives) as something that matched
                matched_diagnoses = self.get_matched_diagnoses(new_potential_diagnoses, self.all_reference_diagnoses)
                alternative_diagnoses = self.get_alternative_diagnoses(matched_diagnoses)
                new_potential_diagnoses = new_potential_diagnoses.union(alternative_diagnoses)
            current_potential_diagnoses = extracted_information['potential diagnoses'][-1] \
                if len(extracted_information['potential diagnoses']) > 0 else set()
            current_potential_diagnoses = sorted(list(
                set(current_potential_diagnoses).union(new_potential_diagnoses)))
            # eliminate anything that matches with a past target
            if len(current_potential_diagnoses) > 0 and len(past_targets) > 0:
                is_match, best_match = self.is_match(current_potential_diagnoses, past_targets)
                current_potential_diagnoses = [d for d, im in zip(current_potential_diagnoses, is_match) if not im]
            extracted_information['potential diagnoses'].append(current_potential_diagnoses)
            # calculate true positives
            if len(extracted_information['potential diagnoses'][-1]) > 0 and \
                    len(extracted_information['target diagnoses'][-1]) > 0:
                is_match, best_match = self.is_match(
                    extracted_information['potential diagnoses'][-1], extracted_information['target diagnoses'][-1])
                true_positives = set([bm for im, bm in zip(is_match, best_match) if im])
            else:
                true_positives = set()
            extracted_information['true positives'].append(true_positives)
        return extracted_information

    def _get_obs(self):
        observed_reports = self._all_reports[self._start_report_index:self._current_report_index + 1]
        evidence = pd.DataFrame(self._current_evidence)
        return {
            "reports": observed_reports.to_csv(index=False),
            "potential_diagnoses": pd.DataFrame({'diagnoses': self._current_potential_diagnoses}).to_csv(index=False),
            "evidence": evidence.to_csv(index=False),
            "evidence_is_retrieved": self._evidence_is_retrieved,
        }

    def is_match(self, xs, ys):
        if self.fuzzy_matching_threshold is not None:
            xs = list(xs)
            ys = list(ys)
            embeddings = self.fuzzy_matching_model.encode(xs + ys, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings[:len(xs)], embeddings[len(xs):])
            indices = cosine_scores.argmax(1)
            return (
                [cosine_scores[i, index] > self.fuzzy_matching_threshold for i, index in enumerate(indices)],
                [ys[index] for index in indices])
        else:
            return [x in ys for x in xs], xs

    def get_matched_diagnoses(self, terms1, terms2):
        if len(terms1) == 0 or len(terms2) == 0:
            return set()
        matched, terms = self.is_match(terms1, tuple(terms2))
        return set([t for m, t in zip(matched, terms) if m])

    def get_alternative_diagnoses(self, diagnoses, symmetric_and_transitive=True):
        all_alternatives = []
        for diagnosis in diagnoses:
            if symmetric_and_transitive:
                all_alternatives.extend([
                    d for i, row in self.alternatives[self.alternatives.full_set.apply(
                        lambda x: diagnosis in x)].iterrows()
                    for d in row.full_set.difference([diagnosis])])
            else:
                alternatives = \
                    self.alternatives[self.alternatives.diagnosis == diagnosis].iloc[0].alternatives.split(',')
                all_alternatives.extend([a.strip() for a in alternatives])
        return set(all_alternatives)

    @property
    def _current_potential_diagnoses(self):
        return self._extracted_information['potential diagnoses'][self._current_report_index]

    @property
    def _current_targets(self):
        targets = self._extracted_information['target diagnoses'][self._current_report_index]
        if self.limit_targets_to is not None:
            targets = [t for t in targets if t in self.limit_targets_to]
        return targets

    def num_examples(self):
        """Returns the number of total instances"""
        if self._all_instances is not None:
            return len(self._all_instances)
        else:
            raise Exception

    def num_unseen_examples(self):
        """Returns the number of novel instances left"""
        if self._all_instances is not None:
            return len(self._all_instances) - len(self._seen_instances)
        else:
            raise Exception

    def reset_epoch(self):
        """Reset seen instances"""
        self._seen_instances = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and 'reports' in options.keys():
            self._all_reports = options['reports']
            self._all_reports['date'] = pd.to_datetime(self._all_reports['date'])
            self._extracted_information = self.extract_info(self._all_reports)
        else:
            assert self._all_instances is not None
            if options is not None and 'instance_index' in options.keys():
                index = options['instance_index']
            else:
                # pick a random novel example
                novel_instances = set(range(len(self._all_instances))).difference(self._seen_instances)
                if len(novel_instances) == 0:
                    print('Gone through all seen examples, resetting epoch!')
                    self.reset_epoch()
                    novel_instances = set(range(len(self._all_instances))).difference(self._seen_instances)
                index = np.random.choice(sorted(list(novel_instances)))
            # add picked example to seen examples
            self._seen_instances.add(index)
            self._all_reports = pd.read_csv(io.StringIO(self._all_instances.iloc[index].reports), parse_dates=['date'])
            if options is not None and 'max_reports_considered' in options.keys():
                self._all_reports = self._all_reports[:options['max_reports_considered']]
            # if indexing into the stored instances use a cache to prevent lots of extraction calls when resetting an
            # environment to a previously seen instance
            if index not in self._extracted_information_cache.keys():
                self._extracted_information_cache[index] = self.extract_info(self._all_reports)
                if self.cache_path is not None:
                    with open(os.path.join(self.cache_path, f'cached_instance_{index}.pkl'), 'wb') as f:
                        pkl.dump({index: self._extracted_information_cache[index]}, f)
            self._extracted_information = self._extracted_information_cache[index]
        # start off with no evidence
        self._current_evidence = {}
        self._evidence_is_retrieved = False
        # TODO: allow one to start the sequence from later on in the ehr
        self._current_report_index = 0
        self._start_report_index = 0
        self.action_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(len(self._current_potential_diagnoses),))
        # go to the first note with a nonzero number of potential diagnoses
        # you should keep increasing the start until you reach at least 2 differential diagnoses
        # but you have to start before the last report
        while len(self._current_potential_diagnoses) < 2 and self._current_report_index + 1 < len(self._all_reports):
            self._current_report_index += 1
            self._start_report_index += 1
            self.action_space = spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(len(self._current_potential_diagnoses),))
        observation = self._get_obs()
        info = {'max_timesteps': (len(self._all_reports) - self._current_report_index) * 2,
                'current_targets': self._current_targets,
                'current_report': self._current_report_index,
                'future_true_positives': set().union(
                    *self._extracted_information['true positives'][self._current_report_index:]),
                'past_targets': self._extracted_information['past target diagnoses'][self._current_report_index]}
        if len(self._current_potential_diagnoses) < 2 and self.verbosity >= 2:
            print('Environment is dead because there is less than 2 differential diagnoses. '
                  'You can either monitor for this by checking with env.is_truncated(obs, info), '
                  'or you can perform a dummy action (which will have no effect) and '
                  'truncate the episode.')
        if len(self._current_targets) < self.num_future_diagnoses_threshold and self.verbosity >= 2:
            print('Environment is dead because there was less than {} extracted targets. '
                  'You can either monitor for this by checking with env.is_truncated(obs, info), '
                  'or you can perform a dummy action (which will have no effect) and '
                  'truncate the episode.'.format(
                self.num_future_diagnoses_threshold))
        if len(info['future_true_positives']) < self.true_positive_minimum and self.verbosity >= 2:
            print('Environment is dead because there was less than {} true positives. '
                  'You can either monitor for this by checking with env.is_truncated(obs, info), '
                  'or you can perform a dummy action (which will have no effect) and '
                  'truncate the episode.'.format(
                self.true_positive_minimum))
        return observation, info

    def _update_evidence(self, diagnoses_to_query):
        for diagnosis in diagnoses_to_query:
            if diagnosis not in self._current_evidence.keys():
                self._current_evidence[diagnosis] = {}
        if 'day' not in self._current_evidence.keys():
            self._current_evidence['day'] = {}
        for i, report_row in self.progress_bar(
                self._all_reports[:self._current_report_index + 1].iterrows(),
                total=self._current_report_index + 1):
            self._current_evidence['day'][i] = (
                report_row.date - self._all_reports.iloc[self._start_report_index].date).days
            text = report_row.text
            # only query ones that have been queried before on this report
            diagnoses_to_query_temp = [x for x in diagnoses_to_query if i not in self._current_evidence[x].keys()]
            if len(diagnoses_to_query_temp) == 0:
                continue
            out = self.model.query(
                tuple(text for _ in diagnoses_to_query_temp),
                tuple(queries['evidence exists'].replace('<evidence query>', diagnosis)
                      for diagnosis in diagnoses_to_query_temp)
            )
            diagnoses_to_query_temp2 = []
            for diagnosis, evidence_exists_out in zip(diagnoses_to_query_temp, out['output']):
                if process_string_output(evidence_exists_out) == 'yes':
                    diagnoses_to_query_temp2.append(diagnosis)
                else:
                    # mark that this query has been made but no evidence was found
                    self._current_evidence[diagnosis][i] = 'no evidence found'
            if len(diagnoses_to_query_temp2) == 0:
                continue
            out = self.model.query(
                tuple(text for _ in diagnoses_to_query_temp2),
                tuple(queries['evidence retrieval'].replace('<evidence query>', diagnosis)
                      for diagnosis in diagnoses_to_query_temp2)
            )
            for diagnosis, evidence in zip(diagnoses_to_query_temp2, out['output']):
                self._current_evidence[diagnosis][i] = evidence

    def reward_per_item(self, action, potential_diagnoses, targets):
        if self.reward_type == 'continuous_independent':
            is_match, best_match = self.is_match(potential_diagnoses, targets)
            reward = action * (torch.tensor(is_match, device=action.device) * 2 - 1)
        elif self.reward_type == 'continuous_dependent':
            is_match, best_match = self.is_match(potential_diagnoses, targets)
            reward = torch.softmax(action, 0)
            reward = reward.masked_fill(~torch.tensor(is_match, device=action.device), 0)
        elif self.reward_type == 'ranking':
            potential_diagnoses = sort_by_scores(potential_diagnoses, action)
            is_match, best_match = self.is_match(potential_diagnoses, targets)
            reward = 1 / (torch.arange(len(is_match)) + 1)
            reward = reward.masked_fill(~torch.tensor(is_match, device=action.device), 0)
        else:
            raise Exception
        return is_match, best_match, reward

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        info = {'max_timesteps': (len(self._all_reports) - self._current_report_index - 1) * 2}
        assert self._current_report_index < len(self._all_reports)
        if not self._evidence_is_retrieved:
            current_potential_diagnoses = sort_by_scores(self._current_potential_diagnoses, action)
            diagnoses_to_query = current_potential_diagnoses[:self.top_k_evidence]
            self._update_evidence(diagnoses_to_query)
            reward = 0
            terminated = False
            self._evidence_is_retrieved = True
        else:
            is_match, best_match, reward = self.reward_per_item(
                action, self._current_potential_diagnoses,
                self._current_targets)
            true_positives = [
                (risk, bm, r)
                for risk, im, bm, r in zip(self._current_potential_diagnoses, is_match, best_match, reward) if im]
            reward = reward.sum().item()
            info['true_positives'] = true_positives
            move_to_next_report = True
            while move_to_next_report:
                self._current_report_index += 1
                self._evidence_is_retrieved = False
                terminated = self._current_report_index == len(self._all_reports) - 1
                if terminated:
                    break
                self.action_space = spaces.Box(
                    low=-float('inf'), high=float('inf'), shape=(len(self._current_potential_diagnoses),))
                move_to_next_report = len(self._current_potential_diagnoses) < 2
        observation = self._get_obs()
        info['current_targets'] = self._current_targets
        info['current_report'] = self._current_report_index
        info['future_true_positives'] = set().union(
            *self._extracted_information['true positives'][self._current_report_index:])
        info['past_targets'] = self._extracted_information['past target diagnoses'][self._current_report_index]
        truncated = self.is_truncated(observation, info)
        return observation, reward, terminated, truncated, info

    def is_truncated(self, obs, info):
        potential_diagnoses = pd.read_csv(io.StringIO(obs['potential_diagnoses'])).diagnoses.to_list()
        return len(potential_diagnoses) < 2 or \
            len(info['current_targets']) < self.num_future_diagnoses_threshold or \
            len(info['future_true_positives']) < self.true_positive_minimum
