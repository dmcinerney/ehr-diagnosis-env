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


# TODO: add a "none of the above" option to the potential diagnoses and the targets?
# TODO: remove all '.' when reading the alternative diagnoses csv


class EHRDiagnosisEnv(gym.Env):
    def __init__(self, instances=None, top_k_evidence=3, llm_name_or_interface='google/flan-t5-xxl',
                 fmm_name_or_interface='all-MiniLM-L6-v2', fuzzy_matching_threshold=.85,
                 progress_bar=None, reward_type='continuous_independent', include_alternative_diagnoses=True,
                 num_future_diagnoses_threshold=1, match_confident_diagnoses=True, match_potential_diagnoses=True,
                 true_positive_minimum=1, cache_path=None, limit_targets_to=None, verbosity=2, alternatives_dir=None,
                 add_risk_factor_queries=True, risk_factors_dir=None, limit_options_with_llm=True):
        """
        :param instances: A dataframe of patient instances with one column of called 'reports' where each element is a
            dataframe of reports ordered by date. The dataframes are in string csv format with one column called 'text'.
        :param top_k_evidence: an int determining how many diagnoses for which to query evidence
        """
        self._all_instances = None
        self._index = None
        self._extracted_information_cache = None
        self._evidence_cache = None
        self.cache_path = None
        if instances is not None:
            self.set_instances(instances, cache_path=cache_path)
        self._seen_instances = set()
        self.top_k_evidence = top_k_evidence
        self.fuzzy_matching_threshold = fuzzy_matching_threshold
        self.progress_bar = progress_bar if progress_bar is not None else tqdm
        self.reward_type = reward_type
        assert self.reward_type in ['continuous_independent', 'continuous_dependent', 'ranking']
        # add addition potential diagnoses that represent alternatives to any of the original ones
        self.include_alternative_diagnoses = include_alternative_diagnoses
        self.num_future_diagnoses_threshold = num_future_diagnoses_threshold
        # restrict confident diagnoses to those that match one of the diagnoses in the alternatives list
        self.match_confident_diagnoses = match_confident_diagnoses
        # restrict potential diagnoses to those that match one of the diagnoses in the alternatives list
        self.match_potential_diagnoses = match_potential_diagnoses
        self.true_positive_minimum = true_positive_minimum
        self.cache_path = cache_path
        self.limit_targets_to = None
        self.set_limit_targets_to(limit_targets_to)
        self.verbosity = verbosity
        self.alternatives_dir = str(files(utils) / 'alternatives') \
            if alternatives_dir is None else alternatives_dir
        self.add_risk_factor_queries = add_risk_factor_queries
        self.risk_factors_dir = str(files(utils) / 'risk_factors') \
            if risk_factors_dir is None else risk_factors_dir
        self.limit_options_with_llm = limit_options_with_llm

        self.seed = None
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'))
        self.observation_space = spaces.Dict(
            {
                # reports seen up to the current timestep in csv format
                "reports": spaces.Text(10000, charset=string.printable),
                # potential diagnoses seen up to the current timestep in csv format
                "options": spaces.Text(10000, charset=string.printable),
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
        self._current_cached_evidence = None

        self._all_reports = None
        self._extracted_information = None

        print('loading model interface')
        if isinstance(llm_name_or_interface, str):
            self.model = get_model_interface(llm_name_or_interface)
        else:
            self.model = llm_name_or_interface
        self.fuzzy_matching_model = None
        if self.fuzzy_matching_threshold is not None:
            if isinstance(fmm_name_or_interface, str):
                self.fuzzy_matching_model = SentenceTransformer(fmm_name_or_interface)
            else:
                self.fuzzy_matching_model = fmm_name_or_interface
        if self.include_alternative_diagnoses or self.match_confident_diagnoses or self.match_potential_diagnoses:
            alternatives_dfs = []
            for file in os.listdir(self.alternatives_dir):
                if file.endswith('.txt'):
                    alternatives_dfs.append(pd.read_csv(os.path.join(self.alternatives_dir, file), delimiter='\t'))
            self.alternatives = pd.concat(alternatives_dfs)
            self.alternatives['full_set'] = self.alternatives.apply(
                lambda r: set([r.diagnosis] + [a.strip() for a in r.alternatives.split(',')]), axis=1)
        self.all_reference_diagnoses = set().union(*self.alternatives.full_set.to_list()) \
            if self.match_confident_diagnoses else None
        if self.add_risk_factor_queries:
            risk_factors_dfs = []
            for file in os.listdir(self.risk_factors_dir):
                if file.endswith('.txt'):
                    risk_factors_dfs.append(pd.read_csv(os.path.join(self.risk_factors_dir, file), delimiter='\t'))
            self.risk_factors = pd.concat(risk_factors_dfs)

    def to(self, device):
        self.model.to(device)
        self.fuzzy_matching_model.to(device)

    def set_instances(self, instances, cache_path=None):
        self._all_instances = instances
        self._extracted_information_cache = {}
        self._evidence_cache = {}
        self.cache_path = cache_path
        if self.cache_path is not None:
            if not os.path.exists(self.cache_path):
                os.mkdir(self.cache_path)
                os.mkdir(os.path.join(self.cache_path, 'extracted_info'))
                os.mkdir(os.path.join(self.cache_path, 'evidence'))
            for file in os.listdir(os.path.join(
                    self.cache_path, 'extracted_info')):
                if file.startswith('cached_instance_'):
                    with open(os.path.join(
                            self.cache_path, 'extracted_info', file), 'rb') \
                            as f:
                        self._extracted_information_cache.update(pkl.load(f))
            for file in os.listdir(os.path.join(
                    self.cache_path, 'evidence')):
                if file.startswith('cached_instance_'):
                    with open(os.path.join(
                            self.cache_path, 'evidence', file), 'rb') \
                            as f:
                        self._evidence_cache.update(pkl.load(f))

    def get_cached_instances(self):
        return self._extracted_information_cache.keys()

    def get_cached_instances_with_queries(self):
        return self._evidence_cache.keys()

    def set_limit_targets_to(self, limit_targets_to):
        self.limit_targets_to = limit_targets_to

    def run_llm_extraction(
            self, text, query_names, post_processing, replace_strings=None):
        # replace strings could be a list of string pairs (2-tuples)
        # that need to be swapped out in the query
        def replace_strings_func(string):
            if replace_strings is not None:
                for s1, s2 in replace_strings:
                    string = string.replace(s1, s2)
            return string
        out = self.model.query(
            tuple(text for _ in range(len(query_names))),
            tuple(replace_strings_func(queries[query_name])
                  for query_name in query_names))
        return [output if func is None else func(output)
                for func, output in zip(post_processing, out['output'])]

    def extract_info(self, reports):
        extracted_information = {
            'differential diagnoses': [],
            'risk prediction': [],
            'presenting complaint': [],
            'differentials from complaint': [],
            'confident diagnoses': [],
        }
        for i, row in self.progress_bar(
                reports.iterrows(), total=len(reports),
                desc='extracting information from reports'):
            first_queries = [
                'differential diagnoses', 'risk prediction',
                'presenting complaint exists', 'confident diagnosis exists']
            post_processing = [
                process_set_output, process_set_output, process_string_output,
                process_string_output]
            dd, rp, pce, cde = self.run_llm_extraction(
                row.text, first_queries, post_processing)
            extracted_information['differential diagnoses'].append(dd)
            extracted_information['risk prediction'].append(rp)
            if pce == 'yes':
                pc, = self.run_llm_extraction(
                    row.text, ['presenting complaint'], [None])
                extracted_information['presenting complaint'].append(
                    process_string_output(pc))
                dfc, = self.run_llm_extraction(
                    '', ['differentials from complaint'], [process_set_output],
                    replace_strings=[('<presenting complaint>', pc)])
                extracted_information['differentials from complaint'].append(
                    dfc)
            else:
                extracted_information['presenting complaint'].append(None)
                extracted_information['differentials from complaint'].append(
                    set())
            if cde == 'yes':
                cd, = self.run_llm_extraction(
                    row.text, ['confident diagnosis'], [None])
                cds, = self.run_llm_extraction(
                    '', ['confident diagnoses extracted'], [process_set_output],
                    replace_strings=[('<confident diagnosis>', cd)])
                extracted_information['confident diagnoses'].append(cds)
            else:
                extracted_information['confident diagnoses'].append(set())
        return extracted_information

    def _process_extracted_info(self, extracted_information):
        processed_extracted_info = {
            'potential diagnoses': [],
            'risk factors': [],
            'target diagnoses': [],
            'target diagnosis countdown': [],
            'true positives': [],
            'past target diagnoses': [],
        }
        confident_diagnoses = []
        confident_diagnosis_timepoints = {}
        for i in range(len(next(iter(extracted_information.values())))):
            extracted_targets = extracted_information['confident diagnoses'][i]
            if self.match_confident_diagnoses:
                extracted_targets = self.get_matched_diagnoses(extracted_targets, self.all_reference_diagnoses)
            confident_diagnoses.append(extracted_targets)
            for target in extracted_targets:
                if target not in confident_diagnosis_timepoints.keys():
                    confident_diagnosis_timepoints[target] = i
        all_targets = set().union(*confident_diagnoses)
        # TODO: add differentials from presenting complaint
        # initially set to the union of all confident diagnoses that emerge in the episode
        for i in range(len(next(iter(extracted_information.values())))):
            differential_diagnoses = extracted_information['differential diagnoses'][i]
            risks = extracted_information['risk prediction'][i]
            # keep a running list of confident diagnoses seen up to and including the current time-points
            previous_past_targets = set() \
                if len(processed_extracted_info['past target diagnoses']) == 0 else \
                processed_extracted_info['past target diagnoses'][-1]
            target_diagnoses = confident_diagnoses[i]
            processed_extracted_info['past target diagnoses'].append(previous_past_targets.union(target_diagnoses))
            # eliminate the targets that appear in the report from the list of targets for that and future time-point
            previous_targets = all_targets \
                if len(processed_extracted_info['target diagnoses']) == 0 else \
                processed_extracted_info['target diagnoses'][-1]
            processed_extracted_info['target diagnoses'].append(previous_targets.difference(target_diagnoses))
            processed_extracted_info['target diagnosis countdown'].append({
                t: confident_diagnosis_timepoints[t] - i
                for t in processed_extracted_info['target diagnoses'][-1]})
            # keep a running list of potential diagnoses by combining differentials and risks
            if self.limit_options_with_llm:
                new_potential_diagnoses = differential_diagnoses.union(risks)
                if self.match_potential_diagnoses:
                    new_potential_diagnoses = self.get_matched_diagnoses(new_potential_diagnoses, self.all_reference_diagnoses)
                if self.include_alternative_diagnoses:
                    # if including alternatives, expand the list by adding anything that appears in the same line (in the
                    # alternatives) as something that matched
                    matched_diagnoses = new_potential_diagnoses if self.match_potential_diagnoses else \
                        self.get_matched_diagnoses(new_potential_diagnoses, self.all_reference_diagnoses)
                    alternative_diagnoses = self.get_alternative_diagnoses(matched_diagnoses)
                    new_potential_diagnoses = new_potential_diagnoses.union(alternative_diagnoses)
                current_potential_diagnoses = processed_extracted_info['potential diagnoses'][-1] \
                    if len(processed_extracted_info['potential diagnoses']) > 0 else set()
                current_potential_diagnoses = sorted(list(
                    set(current_potential_diagnoses).union(new_potential_diagnoses)))
            else:
                current_potential_diagnoses = processed_extracted_info['potential diagnoses'][-1] \
                    if len(processed_extracted_info['potential diagnoses']) > 0 else \
                    self.all_reference_diagnoses
                current_potential_diagnoses = sorted(list(current_potential_diagnoses))
            # eliminate anything that matches with a past target
            if len(current_potential_diagnoses) > 0 and len(processed_extracted_info['past target diagnoses'][-1]) > 0:
                is_match, best_match = self.is_match(
                    current_potential_diagnoses, processed_extracted_info['past target diagnoses'][-1])
                current_potential_diagnoses = [d for d, im in zip(current_potential_diagnoses, is_match) if not im]
            processed_extracted_info['potential diagnoses'].append(current_potential_diagnoses)
            if self.add_risk_factor_queries:
                diagnoses = self.risk_factors.diagnosis.tolist()
                matched_diagnoses = self.get_matched_diagnoses(
                    current_potential_diagnoses, diagnoses)
                risk_factors = set([])
                for d in matched_diagnoses:
                    risk_factors = risk_factors.union(
                        [rf.strip() for rf in
                         self.risk_factors[
                             self.risk_factors['diagnosis'] == d
                             ].iloc[0]['risk factors'].split(',')])
                processed_extracted_info['risk factors'].append(
                    sorted(list(risk_factors)))
            # calculate true positives
            if len(processed_extracted_info['potential diagnoses'][-1]) > 0 and \
                    len(processed_extracted_info['target diagnoses'][-1]) > 0:
                is_match, best_match = self.is_match(
                    processed_extracted_info['potential diagnoses'][-1], processed_extracted_info['target diagnoses'][-1])
                true_positives = set([bm for im, bm in zip(is_match, best_match) if im])
            else:
                true_positives = set()
            processed_extracted_info['true positives'].append(true_positives)
        return processed_extracted_info

    def _get_obs(self):
        observed_reports = self._all_reports[self._start_report_index:self._current_report_index + 1]
        evidence = pd.DataFrame(self._current_evidence)
        return {
            "reports": observed_reports.to_csv(index=False),
            "options": pd.DataFrame({
                'option': self._current_options,
                'type': self._current_option_types}).to_csv(index=False),
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
    def _current_options(self):
        if self._evidence_is_retrieved or not self.add_risk_factor_queries:
            return self._extracted_information[
                'potential diagnoses'][self._current_report_index]
        else:
            return self._extracted_information[
                'potential diagnoses'][self._current_report_index] \
                + self._extracted_information[
                    'risk factors'][self._current_report_index]

    @property
    def _current_option_types(self):
        if self._evidence_is_retrieved or not self.add_risk_factor_queries:
            return ['diagnosis'] * len(self._extracted_information[
                'potential diagnoses'][self._current_report_index])
        else:
            return ['diagnosis'] * len(self._extracted_information[
                'potential diagnoses'][self._current_report_index]) \
                + ['risk factor'] * len(self._extracted_information[
                    'risk factors'][self._current_report_index])

    @property
    def _current_targets(self):
        targets = self._extracted_information['target diagnoses'][
            self._current_report_index]
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
            self._index = None
            self._all_reports = options['reports']
            self._all_reports['date'] = pd.to_datetime(
                self._all_reports['date'])
            self._extracted_information = self.extract_info(self._all_reports)
            self._current_cached_evidence = None
        else:
            assert self._all_instances is not None
            if options is not None and 'instance_index' in options.keys():
                self._index = options['instance_index']
            else:
                # pick a random novel example
                novel_instances = set(range(len(
                    self._all_instances))).difference(self._seen_instances)
                if len(novel_instances) == 0:
                    print('Gone through all seen examples, resetting epoch!')
                    self.reset_epoch()
                    novel_instances = set(range(len(
                        self._all_instances))).difference(self._seen_instances)
                self._index = np.random.choice(sorted(list(novel_instances)))
            # add picked example to seen examples
            self._seen_instances.add(self._index)
            self._all_reports = pd.read_csv(io.StringIO(
                self._all_instances.iloc[self._index].reports),
                parse_dates=['date'])
            if options is not None and \
                    'max_reports_considered' in options.keys():
                self._all_reports = self._all_reports[
                    :options['max_reports_considered']]
            # if indexing into the stored instances use a cache to
            # prevent lots of extraction calls when resetting an
            # environment to a previously seen instance
            if self._index not in self._extracted_information_cache.keys() or \
                    len(self._all_reports) > \
                    len(next(iter(
                    self._extracted_information_cache[self._index].values()))):
                self._extracted_information_cache[self._index] = \
                    self.extract_info(self._all_reports)
                if self.cache_path is not None:
                    with open(os.path.join(
                            self.cache_path,
                            f'extracted_info/cached_instance_{self._index}.pkl'
                            ), 'wb') as f:
                        pkl.dump({
                            self._index: self._extracted_information_cache[
                                self._index]}, f)
            self._extracted_information = \
                self._extracted_information_cache[self._index]
            if self.cache_path is not None:
                self._current_cached_evidence = \
                    self._evidence_cache[self._index] \
                    if self._index in self._evidence_cache.keys() else {}
            else:
                self._current_cached_evidence = None
        self._extracted_information.update(self._process_extracted_info(
            self._extracted_information))
        # start off with no evidence
        self._current_evidence = {}
        self._evidence_is_retrieved = False
        # TODO: allow one to start the sequence from later on in the ehr
        self._current_report_index = 0
        self._start_report_index = 0
        self.action_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(len(self._current_options),))
        # go to the first note with a nonzero number of potential
        # diagnoses, you should keep increasing the start until you
        # reach at least 2 differential diagnoses, but you have to start
        # before the last report
        while len(self._current_options) < 2 and \
                self._current_report_index + 1 < len(self._all_reports):
            self._current_report_index += 1
            self._start_report_index += 1
            self.action_space = spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(len(self._current_options),))
        observation = self._get_obs()
        info = {
            'max_timesteps': (
                len(self._all_reports) - self._current_report_index) * 2,
            'current_targets': self._current_targets,
            'target_countdown':
                self._extracted_information['target diagnosis countdown'][
                    self._current_report_index],
            'current_report': self._current_report_index,
            'future_true_positives': set().union(
                *self._extracted_information['true positives'][
                    self._current_report_index:]),
            'past_targets': self._extracted_information[
                'past target diagnoses'][self._current_report_index]}
        if len([t for t in self._current_option_types if t == 'diagnosis']) \
                < 2 and self.verbosity >= 2:
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

    def _update_evidence(self, query_terms, types):
        for q, t in zip(query_terms, types):
            if (q, t) not in self._current_evidence.keys():
                self._current_evidence[(q, t)] = {}
            if self._current_cached_evidence is not None and \
                    (q, t) not in self._current_cached_evidence.keys():
                self._current_cached_evidence[(q, t)] = {}
        if 'day' not in self._current_evidence.keys():
            self._current_evidence['day'] = {}
        if self._current_cached_evidence is not None and \
                'day' not in self._current_cached_evidence.keys():
            self._current_cached_evidence['day'] = {}
        for i, report_row in self.progress_bar(
                self._all_reports[:self._current_report_index + 1].iterrows(),
                total=self._current_report_index + 1):
            day = (report_row.date - self._all_reports.iloc[
                self._start_report_index].date).days
            self._current_evidence['day'][i] = day
            # fill in any needed evidence that already exists in the cache
            if self._current_cached_evidence is not None:
                if i not in self._current_cached_evidence['day'].keys():
                    self._current_cached_evidence['day'][i] = day
                else:
                    for q, t in zip(query_terms, types):
                        if (q, t) in self._current_cached_evidence.keys() and \
                                i in self._current_cached_evidence[
                                    (q, t)].keys():
                            self._current_evidence[(q, t)][i] = \
                                self._current_cached_evidence[(q, t)][i]
            text = report_row.text
            # only query ones that have not been queried before on this
            # report
            query_terms_temp = [
                x for x in zip(query_terms, types) if i not in self._current_evidence[x].keys()]
            if len(query_terms_temp) == 0:
                continue
            out = self.model.query(
                tuple(text for _ in query_terms_temp),
                tuple(queries['evidence exists'].replace('<evidence query>', q)
                      if t == 'diagnosis' else
                      queries['evidence via rf exists'].replace('<evidence rf query>', q)
                      for q, t in query_terms_temp)
            )
            query_terms_temp2 = []
            for (q, t), evidence_exists_out in zip(query_terms_temp, out['output']):
                if process_string_output(evidence_exists_out) == 'yes':
                    query_terms_temp2.append((q, t))
                else:
                    # mark that this query has been made but no evidence was found
                    self._current_evidence[(q, t)][i] = 'no evidence found'
                    if self._current_cached_evidence is not None:
                        self._current_cached_evidence[(q, t)][i] = 'no evidence found'
            if len(query_terms_temp2) == 0:
                continue
            out = self.model.query(
                tuple(text for _ in query_terms_temp2),
                tuple(queries['evidence retrieval'].replace('<evidence query>', q)
                      if t == 'diagnosis' else
                      queries['evidence via rf retrieval'].replace('<evidence rf query>', q)
                      for q, t in query_terms_temp2)
            )
            for (q, t), evidence in zip(query_terms_temp2, out['output']):
                self._current_evidence[(q, t)][i] = evidence
                if self._current_cached_evidence is not None:
                    self._current_cached_evidence[(q, t)][i] = evidence
        if self._current_cached_evidence is not None:
            self._evidence_cache[self._index] = self._current_cached_evidence
            with open(os.path.join(
                    self.cache_path,
                    f'evidence/cached_instance_{self._index}.pkl'), 'wb') as f:
                pkl.dump({self._index: self._current_cached_evidence}, f)

    def reward_per_item(self, action, potential_diagnoses, targets):
        # To be consistent, all rewards (regardless of reward type) are assumed to be in log space
        # (This will make things easier if numerical instability becomes a problem.)
        if self.reward_type == 'continuous_independent':
            # exp action
            action = torch.exp(action)
            is_match, best_match = self.is_match(potential_diagnoses, targets)
            reward =  action * (torch.tensor(is_match, device=action.device) * 2 - 1)
        elif self.reward_type == 'continuous_dependent':
            # softmax action
            reward = torch.softmax(action, 0)
            is_match, best_match = self.is_match(potential_diagnoses, targets)
            reward = reward.masked_fill(~torch.tensor(is_match, device=action.device), 0)
        elif self.reward_type == 'ranking':
            # no action transformation bc actions is only used for sorting
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
            current_options = sort_by_scores(
                list(zip(self._current_options, self._current_option_types)),
                action)
            query_terms = [q for q, _ in current_options[:self.top_k_evidence]]
            types = [t for _, t in current_options[:self.top_k_evidence]]
            self._update_evidence(query_terms, types)
            reward = 0
            self._evidence_is_retrieved = True
            self.action_space = spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(len(self._current_options),))
        else:
            is_match, best_match, reward = self.reward_per_item(
                action, self._current_options,
                self._current_targets)
            true_positives = [
                (risk, bm, r)
                for risk, im, bm, r in zip(self._current_options, is_match, best_match, reward) if im]
            reward = reward.sum().item()
            info['true_positives'] = true_positives
            move_to_next_report = True
            while move_to_next_report:
                self._current_report_index += 1
                self._evidence_is_retrieved = False
                if self._current_report_index >= len(self._all_reports) - 1:
                    break
                self.action_space = spaces.Box(
                    low=-float('inf'), high=float('inf'), shape=(len(self._current_options),))
                move_to_next_report = len(self._current_options) < 2
        observation = self._get_obs()
        info['current_targets'] = self._current_targets
        info['target_countdown'] = \
            self._extracted_information['target diagnosis countdown'][self._current_report_index]
        info['current_report'] = self._current_report_index
        info['future_true_positives'] = set().union(
            *self._extracted_information['true positives'][self._current_report_index:])
        info['past_targets'] = self._extracted_information['past target diagnoses'][self._current_report_index]
        terminated = self.is_terminated(observation, info)
        truncated = self.is_truncated(observation, info)
        return observation, reward, terminated, truncated, info

    def is_terminated(self, obs, info):
        # the last observation should be of the last report
        # no action is taken on this observation, so is_terminated is
        # true
        return not obs['evidence_is_retrieved'] and \
            info['current_report'] >= len(self._all_reports) - 1

    def is_truncated(self, obs, info):
        options = pd.read_csv(io.StringIO(obs['options']))
        return len(options[options.type == 'diagnosis']) < 2 or \
            len(info['current_targets']) < self.num_future_diagnoses_threshold or \
            len(info['future_true_positives']) < self.true_positive_minimum
