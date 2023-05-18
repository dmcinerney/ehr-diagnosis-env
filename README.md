# ehr-diagnosis-env
A gym environment for diagnosis that uses information extracted EHRs as observations.

Pseudo-code:
```Step 1: extract risk factors/differential diagnoses from all reports, extract confident diagnoses from all reports

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
