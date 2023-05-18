import gymnasium as gym
from gymnasium import spaces
from ehr_diagnosis_env.utils import *
from tqdm import tqdm


class EHRDiagnosisEnv(gym.Env):
    def __init__(self, reports, top_k_evidence=3):
        self._all_reports = reports
        self.top_k_evidence = top_k_evidence

        self.action_space = spaces.Sequence(spaces.Box(low=-float('inf'), high=float('inf')))
        self.observation_space = spaces.Dict(
            {
                # reports seen up to the current timestep
                "reports": spaces.Text(10000),
                # potential diagnoses seen up to the current timestep
                "potential_diagnoses": spaces.Sequence(spaces.Text(10000)),
                # evidence extracted from previous reports for each diagnosis
                "evidence": spaces.Dict(spaces.Sequence(spaces.Text(10000))),
            }
        )

        # denotes the index of the currently observed note, leave to the reset method to set this
        self._current_report_index = None

        # denotes if the action for evidence retrieval has been taken for the currently observed note,
        # leave to the reset method to set this
        self._evidence_is_retrieved = None

        # a running list of currently considered diagnoses to be observed
        self._current_potential_diagnoses = None

        # a running dictionary of retrieved evidence for each diagnosis
        self._current_evidence = None

        self._extracted_information = {
            'differential diagnoses': [],
            'risk prediction': [],
            'confident diagnoses': [],
        }
        print('loading model interface')
        self.model = get_model_interface('google/flan-t5-xxl')
        for i, row in tqdm(
                self._all_reports.iterrows(), total=len(self._all_reports), desc='extracting information from reports'):
            first_queries = ['differential diagnoses', 'risk prediction', 'confident diagnosis exists']
            out = self.model.query(
                tuple(row.text for _ in range(len(first_queries))),
                tuple(queries[query_name] for query_name in first_queries))
            self._extracted_information['differential diagnoses'].append(process_set_output(out['output'][0]))
            self._extracted_information['risk prediction'].append(process_set_output(out['output'][1]))
            confident_diagnosis_exists = process_string_output(out['output'][2])
            if confident_diagnosis_exists == 'yes':
                out = self.model.query((row.text,), (queries['confident diagnosis']))
                out = self.model.query(
                    ('',),
                    (queries['confident diagnoses extracted'].replace('<confident diagnosis>', out['output'][0])))
                self._extracted_information['confident diagnoses'].append(process_set_output(out['output'][0]))
            else:
                self._extracted_information['confident diagnoses'].append(set())
        # initially set to the union of all confident diagnoses that appear in the future
        self._current_targets = set().union(*self._extracted_information['confident diagnoses'])

    def _get_obs(self):
        return {
            "reports": self._all_reports[:self._current_report_index + 1].to_csv(index=False),
            "potential_diagnoses": self._current_potential_diagnoses,
            "evidence": self._current_evidence,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_report_index = 0
        self._evidence_is_retrieved = False
        # start off with the extractions from the first note
        self._current_potential_diagnoses = sorted(list(set().union(
            self._extracted_information['differential diagnoses'] + self._extracted_information['risk prediction'])))
        # start off with no evidence
        self._current_evidence = {}
        # delete the known confident diagnoses given the first report from the targets
        self._current_targets = self._current_targets.difference(
            self._extracted_information['confident diagnoses'][self._current_report_index])
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        reward, terminated, truncated = None, None, None

        if not self._evidence_is_retrieved:
            self._current_potential_diagnoses = sort_by_scores(self._current_potential_diagnoses, action)
            new_evidence = {}
            out = self.model.query(
                tuple(self._all_reports.iloc[self._current_report_index].text for _ in range(self.top_k_evidence)),
                tuple(queries['evidence exists'].replace('<evidence query>', diagnosis)
                      for diagnosis in self._current_potential_diagnoses[:self.top_k_evidence])
            )
                for
                if process_string_output(out['output'][0])
            self._current_evidence = {}
            # delete the known confident diagnoses given the first report from the targets
            self._current_targets = self._current_targets.difference(
                self._extracted_information['confident diagnoses'][self._current_report_index])

        else:
            self._current_report_index += 1
            self._current_potential_diagnoses = sorted(list(set(self._current_potential_diagnoses).union(
                self._extracted_information['differential diagnoses'][self._current_report_index] +
                self._extracted_information['risk prediction'][self._current_report_index])))

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
