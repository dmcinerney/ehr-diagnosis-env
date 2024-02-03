from gymnasium.envs.registration import register


register(
     id="ehr_diagnosis_env/EHRDiagnosisEnv-v0",
     entry_point="ehr_diagnosis_env.envs:EHRDiagnosisEnv",
)
register(
     id="ehr_diagnosis_env/FlanEHRDiagnosisEnv-v0",
     entry_point="ehr_diagnosis_env.envs:FlanEHRDiagnosisEnv",
)
register(
     id="ehr_diagnosis_env/MistralEHRDiagnosisEnv-v0",
     entry_point="ehr_diagnosis_env.envs:MistralEHRDiagnosisEnv",
)
