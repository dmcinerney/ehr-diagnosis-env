from .postprocessing import *
from .query import *


prompt = """Read the following report:

<input>

Provide a list of the considered differential diagnoses. """
registered_queries[('flan', 'differential diagnoses')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

Provide a list of diagnoses not yet considered that should be. """
registered_queries[('flan', 'other diagnoses')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """A doctor has provided the following list of potential diagnoses:

<current differential diagnoses>

Not including these, provide a list of alternative diagnoses that should be considered if any exist: """
registered_queries[('flan', 'other diagnoses 2')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<current differential diagnoses>')
    + len('<current differential diagnoses>'))


prompt = """Read the following report:

<input>

Question: Is there a confident diagnosis of the patient's condition? Choice: -Yes -No
Answer: """
registered_queries[('flan', 'confident diagnosis exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

Answer step by step: What is the correct diagnosis of the patient's condition?
Answer: """
registered_queries[('flan', 'confident diagnosis')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Here is a diagnosis of a patient:

<confident diagnosis>

Question: Provide a list of diagnostic terms or write none.
Answer: """
registered_queries[('flan', 'confident diagnoses extracted')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<confident diagnosis>')
    + len('<confident diagnosis>'))


prompt = """Here an admission note:

<input>

Question: Based on this note, provide a list of conditions the patient may be at risk for.
Answer: """
registered_queries[('flan', 'risk prediction')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following clinical note of a patient:

<input>

Question: Is the patient at risk of <query>? Choice: -Yes -No
Answer: """
registered_queries[('flan', 'evidence exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following clinical note of a patient:

<input>

Based on the note, why is the patient at risk of <query>?
Answer step by step: """
registered_queries[('flan', 'evidence retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following clinical note of a patient:

<input>

Question: Does the patient have <query>? Choice: -Yes -No
Answer: """
registered_queries[('flan', 'evidence has condition exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following clinical note of a patient:

<input>

Question: Extract signs of <query> from the note.
Answer: """
registered_queries[('flan', 'evidence has condition retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

Question: Does the patient have <query>? Choice: -Yes -No
Answer: """
registered_queries[('flan', 'evidence via rf exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

What evidence is there that the patient has <query>?
Answer: """
registered_queries[('flan', 'evidence via rf retrieval')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

Does this report contain the presenting complaint?
Answer: """
registered_queries[('flan', 'presenting complaint exists')] = Query(
    prompt, postprocessing=lambda rd, gk: process_yes_no_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """Read the following report:

<input>

What is the patient's presenting complaint?
Answer: """
registered_queries[('flan', 'presenting complaint')] = Query(
    prompt, postprocessing=lambda rd, gk: process_string_output(rd['output']),
    truncation_index=prompt.index('<input>') + len('<input>'))


prompt = """A patient presents with <presenting complaint>. Provide a list of the differential diagnoses a clinician should consider.
Answer: """
registered_queries[('flan', 'differentials from complaint')] = Query(
    prompt, postprocessing=lambda rd, gk: process_set_output(rd['output']),
    truncation_index=prompt.index('<presenting complaint>')
    + len('<presenting complaint>'))
