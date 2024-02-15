# EHR Diagnosis Environment
This repository contains a gym environment for a diagnosis risk prediction task that uses information extracted from EHRs as observations and to evaluate reward. An example of usage and training an agent can be seen here:

https://github.com/dmcinerney/ehr-diagnosis-agent

This example repository also contains code for preprocessing MIMIC-III (https://physionet.org/content/mimiciii/1.4/) into the correct format for this environment and it can be easily extended to other EHR sources.

This environment as well as any trained agents can also be visualized using the interface here:

https://github.com/dmcinerney/ehr-diagnosis-env-interface

Paper and Detailed usage instructions coming soon.


### Pseudo-code
```
Step 1: extract risk factors/differential diagnoses from all reports, extract confident diagnoses from all reports

Step 2: episode rollout:
	For each report in sequential order:
		Step 1: predict top k risk factors/differential diagnoses to query given the unordered risk factors/differential diagnoses from all reports through the current
		Step 2: query these for all previous reports to get evidence and add to the running list of evidence
		Step 3: rerank these based on the evidence from all previous queries
		Step 4: evaluate ranked risk factors/differential diagnoses against the future confident diagnoses with the reward function

Reward function for one timestep = 
	Sum for all risk factors/differential diagnoses of:
		{
			0 if not in future diagnoses
			1/rank if in future diagnoses (do some fuzzy matching, maybe thresholded cosine sim)
		}
```

TODO: Add dummy data for example.

