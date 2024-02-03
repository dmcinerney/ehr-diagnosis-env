from .ehr_diagnosis_env import EHRDiagnosisEnv
from ehr_diagnosis_env.utils import *


class FlanEHRDiagnosisEnv(EHRDiagnosisEnv):
    def __init__(
            self,
            *,
            llm_name_or_interface='google/flan-t5-xxl',
            **kwargs
        ):
        super().__init__(
            llm_name_or_interface=llm_name_or_interface,
            **kwargs
        )

    @property
    def llm_model_type(self):
        return 'flan'

    def extract_info_from_report_row(self, row):
        """
        Should return a dictionary containing at least:
        - differential diagnoses: set
        - confident diagnoses: set
        """
        extracted_information = {
            'differential diagnoses': set(),
            'confident diagnoses': set(),
            'original differential diagnoses': set(),
            'risk prediction': set(),
            'presenting complaint': None,
            'differentials from complaint': set(),
        }
        first_queries = [
            'differential diagnoses', 'risk prediction',
            'presenting complaint exists', 'confident diagnosis exists']
        confident_diagnosis_text = confident_diagnosis_preprocessing(
            row.text)
        texts = [
            row.text, row.text, row.text, confident_diagnosis_text]
        dd, rp, pce, cde = self.run_llm_extraction(
            first_queries, [{'input': t} for t in texts])['processed_output']
        extracted_information['original differential diagnoses'] = dd
        extracted_information['risk prediction'] = rp
        extracted_information['differential diagnoses'] = dd.union(rp)
        if pce:
            pc, = self.run_llm_extraction(
                ['presenting complaint'], [{'input': row.text}])['processed_output']
            extracted_information['presenting complaint'] = pc
            dfc, = self.run_llm_extraction(
                ['differentials from complaint'],
                [{'input': '', 'presenting complaint': pc}]
                )['processed_output']
            extracted_information['differentials from complaint'] = dfc
            extracted_information['differential diagnoses'] = \
                extracted_information['differential diagnoses'].union(dfc)
        if cde:
            cd, = self.run_llm_extraction(
                ['confident diagnosis'],
                [{'input': confident_diagnosis_text}])['processed_output']
            cds, = self.run_llm_extraction(
                ['confident diagnoses extracted'],
                [{'input': '', 'confident diagnosis': cd}]
                )['processed_output']
            extracted_information['confident diagnoses'] = cds
        return extracted_information
