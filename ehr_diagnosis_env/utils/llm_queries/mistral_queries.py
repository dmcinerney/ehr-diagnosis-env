from .postprocessing import *
from .query import *
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
parenthesis_ids = tokenizer([' ('], add_special_tokens=False).input_ids


# def mistral_confident_diagnoses(yaml_output):
#     if yaml_output is None:
#         return set()
#     definite_diagnoses = set()
#     for dictionary in yaml_output:
#         if isinstance(dictionary, dict) and \
#                 'diagnosis' in dictionary.keys() and \
#                 isinstance(dictionary['diagnosis'], str) and \
#                 'certainty' in dictionary.keys() and \
#                 str(dictionary['certainty']).lower().strip() in [
#                     'definite', 'certain'] and \
#                 'is_current' in dictionary.keys() and \
#                 isinstance(dictionary['is_current'], bool) and \
#                 dictionary['is_current']:
#             definite_diagnoses.add(
#                 dictionary['diagnosis'].split('(')[0].lower().strip())
#     return definite_diagnoses
# def mistral_unconfident_diagnoses(yaml_output):
#     if yaml_output is None:
#         return set()
#     unconfident_diagnoses = set()
#     for dictionary in yaml_output:
#         if isinstance(dictionary, dict) and \
#                 'diagnosis' in dictionary.keys() and \
#                 isinstance(dictionary['diagnosis'], str) and \
#                 'certainty' in dictionary.keys() and \
#                 not str(dictionary['certainty']).lower().strip() in [
#                     'definite', 'certain']:
#             unconfident_diagnoses.add(
#                 dictionary['diagnosis'].split('(')[0].lower().strip())
#     return unconfident_diagnoses
# prompt = """<s>[INST] Here is a report from a patient's medical record:

# <input>

# Provide a concise list of the diagnoses (e.g. pneumonia, pulmonary edema, lung cancer, etc.), their corresponding certainty (e.g. uncertain, low, medium, high, definite), and if they are current or historical diagnoses in a valid yaml format.

# Format example:
# ```yaml
# - diagnosis: <diagnosis 1>
#   certainty: <certainty 1>
#   is_current: <is_current 1>
# - diagnosis: <diagnosis 2>
#   certainty: <certainty 2>
#   is_current: <is_current 2>
# ``` [/INST] """
# registered_queries[('mistral', 'diagnoses')] = Query(
#     prompt, truncation_index=prompt.index('<input>') + len('<input>'),
#     generation_kwargs={'max_new_tokens': 256})



def mistral_confident_diagnoses(yaml_output):
    if yaml_output is None or not isinstance(yaml_output, list):
        return set()
    definite_diagnoses = set()
    for x in yaml_output:
        if not isinstance(x, dict) or len(x) != 1:
            continue
        diagnosis, confidence = next(iter(x.items()))
        if isinstance(diagnosis, str) and isinstance(confidence, str) and \
                confidence.lower().strip() in ['definite', 'certain']:
            definite_diagnoses.add(diagnosis.lower().strip())
    return definite_diagnoses
def mistral_unconfident_diagnoses(yaml_output):
    if yaml_output is None or not isinstance(yaml_output, list):
        return set()
    unconfident_diagnoses = set()
    for x in yaml_output:
        if not isinstance(x, dict) or len(x) != 1:
            continue
        diagnosis, confidence = next(iter(x.items()))
        if isinstance(diagnosis, str) and isinstance(confidence, str) and \
                not confidence.lower().strip() in ['definite', 'certain']:
            unconfident_diagnoses.add(diagnosis.lower().strip())
    return unconfident_diagnoses
prompt = """<s>[INST] Here is a report from a patient's medical record:

<input>

Provide a concise list of the diagnoses (e.g. pneumonia, pulmonary edema, lung cancer, etc.) and their corresponding certainty (e.g. uncertain, low, medium, high, definite) in a valid yaml format.

Format example:
```yaml
- <diagnosis 1>: <certainty 1>
- <diagnosis 2>: <certainty 2>
...
``` [/INST] """
registered_queries[('mistral', 'diagnoses')] = Query(
    prompt, truncation_index=prompt.index('<input>') + len('<input>'),
    generation_kwargs={
        'max_new_tokens': 256,
        'bad_word_ids': parenthesis_ids}) # not sure bad_word_ids is actually working


def mistral_differentials(yaml_output):
    if yaml_output is None or not isinstance(yaml_output, list):
        return set()
    differentials = set()
    for x in yaml_output:
        if isinstance(x, str):
            differentials.add(x)
        elif isinstance(x, dict) and len(x) == 1:
            differentials.add(next(iter(x.keys())))
    return differentials
prompt = """<s>[INST] Here is a report from a patient's medical record:

<input>

Provide a concise list of differential diagnoses that should be considered for this patient in a valid yaml format.

Format example:
```yaml
- <diagnosis 1>
- <diagnosis 2>
...
``` [/INST] """
registered_queries[('mistral', 'differentials')] = Query(
    prompt, postprocessing=lambda rd, gk: mistral_differentials(
        yaml_postprocess(truncate_if_ends_early(rd))),
    truncation_index=prompt.index('<input>') + len('<input>'),
    generation_kwargs={
        'max_new_tokens': 256,
        'bad_word_ids': parenthesis_ids})








prompt = """<s>[INST] Read the following clinical note of a patient:

<input>

Question: Is the patient at risk of <query>? Choice: -Yes -No
Answer: [/INST] """
registered_queries[('mistral', 'evidence exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """<s>[INST] Read the following clinical note of a patient:

<input>

Based on the note, why is the patient at risk of <query>? Be concise.
Answer: [/INST] """
registered_queries[('mistral', 'evidence retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """<s>[INST] Read the following clinical note of a patient:

<input>

Question: Does the patient have <query>? Choice: -Yes -No
Answer: [/INST] """
registered_queries[('mistral', 'evidence has condition exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """<s>[INST] Read the following clinical note of a patient:

<input>

Question: Extract signs of <query> from the note. Be concise.
Answer: [/INST] """
registered_queries[('mistral', 'evidence has condition retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """<s>[INST] Read the following report:

<input>

Question: Does the patient have <query>? Choice: -Yes -No
Answer: [/INST] """
registered_queries[('mistral', 'evidence via rf exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """<s>[INST] Read the following report:

<input>

What evidence is there that the patient has <query>? Be concise.
Answer: [/INST] """
registered_queries[('mistral', 'evidence via rf retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))
