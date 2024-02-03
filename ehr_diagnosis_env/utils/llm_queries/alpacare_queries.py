from .postprocessing import *
from .query import *


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the following report from a patient's medical record, what are the definitive diagnoses of the patient if any exist? Please only include this list and not any explanation.

### Input:
<input>

### Response:"""
registered_queries[('alpacare', 'confident diagnoses')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the following report from a patient's medical record, provide a list of the differential diagnoses that should be considered. Please only include this list and not any explanation.

### Input:
<input>

### Response:"""
registered_queries[('alpacare', 'differentials')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


# prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# Given the following report from a patient's medical record, provide a list of symptoms the patient exhibits. Please only include this list and not any explanation.

# ### Input:
# <input>

# ### Response:"""
# registered_queries[('alpacare', 'symptoms')] = Query(
#     prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
#     truncation_index=prompt.index('<input>') + len('<input>'))
