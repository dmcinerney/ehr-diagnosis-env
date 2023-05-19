import gymnasium as gym
from gymnasium import spaces
from ehr_diagnosis_env.utils import *
from tqdm import tqdm
import pandas as pd
import io
import string
from sentence_transformers import SentenceTransformer, util


class EHRDiagnosisEnv(gym.Env):
    def __init__(self, instances=None, top_k_evidence=3, model_name='google/flan-t5-xxl', fuzzy_matching_threshold=.75):
        """
        :param instances: A dataframe of patient instances with one column of called 'reports' where each element is a
            dataframe of reports ordered by date. The dataframes are in string csv format with one column called 'text'.
        :param top_k_evidence: an int determining how many diagnoses for which to query evidence
        """
        self._all_instances = instances
        self.top_k_evidence = top_k_evidence
        self.model_name = model_name
        self.fuzzy_matching_threshold = fuzzy_matching_threshold

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
            }
        )

        # denotes the index of the currently observed note, leave to the reset method to set this
        self._current_report_index = None

        # denotes if the action for evidence retrieval has been taken for the currently observed note,
        # leave to the reset method to set this
        self._evidence_is_retrieved = None

        # a running list of currently considered diagnoses to be observed
        self._current_potential_diagnoses = None

        # a running dictionary of dictionaries keeping track of retrieved evidence for each diagnosis for each report
        self._current_evidence = None

        self._all_reports = None
        self._extracted_information = None
        self._current_targets = None
        print('loading model interface')
        self.model = get_model_interface(model_name)
        self.fuzzy_matching_model = None
        if self.fuzzy_matching_threshold is not None:
            self.fuzzy_matching_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_info(self, reports):
        extracted_information = {
            'differential diagnoses': [],
            'risk prediction': [],
            'confident diagnoses': [],
        }
        for i, row in tqdm(reports.iterrows(), total=len(reports), desc='extracting information from reports'):
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
        # initially set to the union of all confident diagnoses that appear in the future
        current_targets = set().union(*extracted_information['confident diagnoses'])
        return extracted_information, current_targets

    def _get_obs(self):
        return {
            "reports": self._all_reports[:self._current_report_index + 1].to_csv(index=False),
            "potential_diagnoses": pd.DataFrame({'diagnoses': self._current_potential_diagnoses}).to_csv(index=False),
            "evidence": pd.DataFrame(self._current_evidence).to_csv(index=False),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and 'reports' in options.keys():
            self._all_reports = options['reports']
        else:
            assert self._all_instances is not None
            self._all_reports = pd.read_csv(io.StringIO(self._all_instances.sample(n=1).iloc[0].reports))
        self._extracted_information, self._current_targets = self.extract_info(self._all_reports)
        self._current_report_index = 0
        self._evidence_is_retrieved = False
        # start off with the extractions from the first note
        self._current_potential_diagnoses = sorted(list(set().union(
            self._extracted_information['differential diagnoses'][self._current_report_index],
            self._extracted_information['risk prediction'][self._current_report_index])))
        self.action_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(len(self._current_potential_diagnoses),))
        # start off with no evidence
        self._current_evidence = {}
        # delete the known confident diagnoses given the first report from the targets
        self._current_targets = self._current_targets.difference(
            self._extracted_information['confident diagnoses'][self._current_report_index])
        observation = self._get_obs()
        info = {'max_timesteps': len(self._all_reports) * 2, 'current_targets': self._current_targets,
                'current_report': 0, 'evidence_is_retrieved': False}
        if len(self._current_targets) == 0:
            print('Environment is dead because there were no extracted targets. '
                  'You can either monitor for this by checking the length of the current targets in info, or '
                  'you can perform a dummy action (which will have no effect) and truncate the episode.')
        return observation, info

    def _update_evidence(self, diagnoses_to_query):
        for diagnosis in diagnoses_to_query:
            if diagnosis not in self._current_evidence.keys():
                self._current_evidence[diagnosis] = {}
        for i, report_row in tqdm(
                self._all_reports[:self._current_report_index + 1].iterrows(),
                total=self._current_report_index + 1):
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

    def is_match(self, x, ys):
        if self.fuzzy_matching_threshold is not None:
            ys = list(ys)
            embeddings = self.fuzzy_matching_model.encode([x] + ys, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings[:1], embeddings[1:])
            index = cosine_scores[0].argmax().item()
            if cosine_scores[0, index] > self.fuzzy_matching_threshold:
                return True, ys[index]
            else:
                return False, None
        else:
            return x in ys, x

    def step(self, action):
        info = {'max_timesteps': len(self._all_reports) * 2}
        assert self._current_report_index < len(self._all_reports)
        self._current_potential_diagnoses = sort_by_scores(self._current_potential_diagnoses, action)
        if not self._evidence_is_retrieved:
            diagnoses_to_query = self._current_potential_diagnoses[:self.top_k_evidence]
            self._update_evidence(diagnoses_to_query)
            reward = 0
            terminated = False
            self._evidence_is_retrieved = True
        else:
            reward = 0
            true_positives = []
            for j, risk in enumerate(self._current_potential_diagnoses):
                is_match, best_match = self.is_match(risk, self._current_targets)
                if is_match:
                    reward += 1 / (j + 1)
                    true_positives.append((risk, best_match, j + 1))
            info['true_positives'] = true_positives
            self._current_report_index += 1
            self._evidence_is_retrieved = False
            terminated = self._current_report_index == len(self._all_reports)
            if not terminated:
                # delete the known confident diagnoses given the first report from the targets
                self._current_targets = self._current_targets.difference(
                    self._extracted_information['confident diagnoses'][self._current_report_index])
                self._current_potential_diagnoses = sorted(list(set(self._current_potential_diagnoses).union(
                    self._extracted_information['differential diagnoses'][self._current_report_index],
                    self._extracted_information['risk prediction'][self._current_report_index])))
                self.action_space = spaces.Box(
                    low=-float('inf'), high=float('inf'), shape=(len(self._current_potential_diagnoses),))
        observation = self._get_obs()
        info['current_targets'] = self._current_targets
        info['current_report'] = self._current_report_index
        info['evidence_is_retrieved'] = self._evidence_is_retrieved
        truncated = len(self._current_targets) == 0
        return observation, reward, terminated, truncated, info
