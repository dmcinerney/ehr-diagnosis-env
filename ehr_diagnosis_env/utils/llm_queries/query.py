class Query:
    def __init__(self, template, postprocessing=None, truncation_index=None,
                 generation_kwargs=None):
        self.template = template
        self.postprocessing = postprocessing
        self.truncation_index = truncation_index
        self.generation_kwargs = {} if generation_kwargs is None else \
            generation_kwargs

    def get_inputs(self, **template_values):
        if self.truncation_index is None:
            truncatable = self.template
            fixed = ""
        else:
            truncatable = self.template[:self.truncation_index]
            fixed = self.template[self.truncation_index:]
        for key, value in template_values.items():
            truncatable = truncatable.replace(f'<{key}>', value)
            fixed = fixed.replace(f'<{key}>', value)
        input_text = truncatable + fixed
        only_truncate_prefix = len(truncatable)
        return input_text, only_truncate_prefix

    def get_outputs(self, return_dict, idx):
        new_return_dict = {
            k: v[idx] for k, v in return_dict.items()
            if k != 'generation_kwargs'}
        new_return_dict['generation_kwargs'] = return_dict['generation_kwargs']
        output = new_return_dict['output']
        if self.postprocessing is not None:
            new_return_dict['processed_output'] = self.postprocessing(
                new_return_dict, self.generation_kwargs)
        else:
            new_return_dict['processed_output'] = output
        return new_return_dict

    def __call__(self, model_interface, **template_values):
        input_text, only_truncate_prefix = self.get_inputs(**template_values)
        return_dict = model_interface.query(
            [input_text], only_truncate_prefix=[only_truncate_prefix],
            **self.generation_kwargs)
        return self.get_outputs(return_dict, 0)


class BatchedQueries:
    def __init__(self, *queries):
        self.queries = queries
        assert len(self.queries) > 0
        # TODO: check if the generation kwargs line up! If they don't,
        #   the model will currently generate according to the first queries
        #   generation kwargs

    def __call__(self, model_interface, inputs):
        # Here, inputs should be a list of dictionaries
        # where each dictionary is a set of template_values
        assert len(inputs) == len(self.queries)
        input_texts, only_truncate_prefixes = [], []
        for q, template_values in zip(self.queries, inputs):
            input_text, only_truncate_prefix = q.get_inputs(**template_values)
            input_texts.append(input_text)
            only_truncate_prefixes.append(only_truncate_prefix)
        return_dict = model_interface.query(
            input_texts, only_truncate_prefix=only_truncate_prefixes,
            **self.queries[0].generation_kwargs)
        return [
            q.get_outputs(return_dict, i) for i, q in enumerate(self.queries)]


registered_queries = {}


# Example query:
# def mistral_confident_diagnoses(output):
#     output_yaml = yaml_postprocess(output)
#     definite_diagnoses = set()
#     for dictionary in output_yaml:
#         if 'diagnosis' in dictionary.keys() and \
#                 'certainty' in dictionary.keys() and \
#                 dictionary['certainty'].lower().strip() == 'definite':
#             definite_diagnoses.add(dictionary['diagnosis'].lower().strip())
#     return definite_diagnoses
# prompt = """<s>[INST]Here is a report from a patient's medical record:

# <input>

# Provide a list of the diagnoses (e.g. pneumonia, pulmonary edema, lung cancer, etc.) and their corresponding certainty (e.g. uncertain, low, medium, high, definite) in a valid yaml format.

# Format example:
# ```yaml
# - diagnosis: <diagnosis 1>
#   certainty: <certainty 1>
# - diagnosis: <diagnosis 2>
#   certainty: <certainty 2>
# ```[/INST]"""
# registered_queries[('mistral', 'confident_diagnoses')] = Query(
#     prompt, postprocessing=mistral_confident_diagnoses,
#     truncation_index=prompt.index('<input>') + 7)
