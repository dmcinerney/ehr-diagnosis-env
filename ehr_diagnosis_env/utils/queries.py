import re


queries = {
    'differential diagnoses': \
"""Read the following report:

<input>

Provide a list of the considered differential diagnoses. """,
    'other diagnoses': \
"""Read the following report:

<input>

Provide a list of diagnoses not yet considered that should be. """,
    'other diagnoses 2': \
"""A doctor has provided the following list of potential diagnoses:

<current differential diagnoses>

Not including these, provide a list of alternative diagnoses that should be considered if any exist: """,
    'confident diagnosis exists': \
"""Read the following report:

<input>

Question: Is there a confident diagnosis of the patient's condition? Choice: -Yes -No
Answer: """,
    'confident diagnosis': \
"""Read the following report:

<input>

Answer step by step: What is the correct diagnosis of the patient's condition?

Answer: """,
    'confident diagnoses extracted': \
"""Here is a diagnosis of a patient:

<confident diagnosis>

Question: Based on this diagnosis, provide a list of diagnostic terms.

Answer: """,
    'risk prediction': \
"""Here an admission note:

<input>

Question: Based on this note, provide a list of conditions the patient may be at risk for.

Answer: """,
    'evidence exists': \
"""Read the following report:

<input>

Question: Is the patient at risk of <evidence query>? Choice: -Yes -No
Answer: """,
    'evidence retrieval': \
"""Read the following report:

<input>

Answer step by step: Why is the patient at risk of <evidence query>?
Answer: """,
}


def process_string_output(output):
    # strip, lowercase, delete all non-alphanumeric characters, and standardize all whitespaces
    return ' '.join(re.sub('[\W_]+', ' ', output.strip().lower()).split())


def process_set_output(output):
    # decide what to consider a delimiter
    delimiter = ' -' if ' -' in output else ';' if ';' in output else ','
    output = ' ' + output
    # split string by delimiter
    output = set([process_string_output(x) for x in output.split(delimiter)])
    # delete empty strings and substrings of those that already appear in the set
    output = set([x for x in output if x != '' and all([not x in y for y in output - {x}])])
    # if none is the only element, return the empty set
    if len(output) == 1 and list(output)[0] == 'none':
        return set()
    return output
