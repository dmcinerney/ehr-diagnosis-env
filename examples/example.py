import ehr_diagnosis_env
import gymnasium
import pandas as pd
import io
import time


instance_num = 0
# instance_num = 1
instances = pd.read_csv(
    '/work/frink/mcinerney.de/datasets/mimic-iii/physionet.org/files/mimiciii/1.4/preprocessed/reports_and_codes3/val.data',
    compression='gzip')
env = gymnasium.make(
    'ehr_diagnosis_env/EHRDiagnosisEnv-v0', instances=instances[instance_num:instance_num+1],
    # model_name='google/flan-t5-xl'
    model_name='google/flan-t5-xxl'
)
observation, info = env.reset()
cumulative_reward = 0
for i in range(1000):
    print(f'Timestep {i + 1}')
    for k, v in observation.items():
        print(k.upper())
        try:
            print(pd.read_csv(io.StringIO(v)))
        except Exception:
            print(v)
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print('ACTION')
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print('INFO', info)
    print('REWARD', reward)
    cumulative_reward += reward
    print('CUMULATIVE REWARD', cumulative_reward)
    print('TERMINATED', terminated)
    print('TRUNCATED', truncated)
    if terminated or truncated:
        break
    time.sleep(5) # so you can see the output before the next timestep
    print('\n\n\n\n')
env.close()
