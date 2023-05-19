from gymnasium.envs.registration import register


register(
     id="ehr_diagnosis_env/EHRDiagnosisEnv-v0",
     entry_point="ehr_diagnosis_env.envs:EHRDiagnosisEnv",
)
