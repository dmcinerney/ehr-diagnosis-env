from .ehr_diagnosis_env import EHRDiagnosisEnv
from ehr_diagnosis_env.utils import *


class MistralEHRDiagnosisEnv(EHRDiagnosisEnv):
    def __init__(
            self,
            *,
            llm_name_or_interface='mistralai/Mistral-7B-Instruct-v0.2',
            # llm_max_batch_size=1,
            **kwargs
        ):
        super().__init__(
            llm_name_or_interface=llm_name_or_interface,
            **kwargs
        )

    @property
    def llm_model_type(self):
        return 'mistral'

    def extract_info_from_report_row(self, row):
        """
        Should return a dictionary containing at least:
        - differential diagnoses: set
        - confident diagnoses: set
        """
        # first_queries = ['diagnoses', 'differentials']
        first_queries = ['diagnoses']
        # confident_diagnosis_text = confident_diagnosis_preprocessing(
        #     row.text)
        # texts = [row.text, row.text]
        texts = [row.text]
        outs = self.run_llm_extraction(
            first_queries, [{'input': t} for t in texts])
        # differentials = outs['processed_output'][1]
        diagnoses = yaml_postprocess(truncate_if_ends_early({
            k: v[0] for k, v in outs.items()}))
        if diagnoses is not None:
            confident_diagnoses = mistral_confident_diagnoses(diagnoses)
            unconfident_diagnoses = mistral_unconfident_diagnoses(diagnoses)
        else:
            confident_diagnoses = set()
            unconfident_diagnoses = set()
        extracted_information = {
            'differential diagnoses':
                # unconfident_diagnoses.union(differentials),
                unconfident_diagnoses,
            'confident diagnoses': set(confident_diagnoses),
        }
        return extracted_information
