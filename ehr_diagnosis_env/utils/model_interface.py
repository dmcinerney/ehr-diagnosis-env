from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class ModelInterface:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = self.get_tokenizer(model_name)
        self.model = self.get_model(model_name)
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def get_tokenizer(self, model_name):
        raise NotImplementedError

    def get_model(self, model_name):
        raise NotImplementedError

    def query(self, inputs, prompts, max_new_tokens=50):
        assert len(inputs) == len(prompts)
        prefixes = []
        postfixes = []
        for prompt in prompts:
            if '<input>' in prompt:
                prefix, postfix = prompt.split('<input>')
            else:
                prefix, postfix = None, prompt
            prefixes.append(prefix)
            postfixes.append(postfix)
        # TODO: include logic for adding more than one chunk per input
        tokenized_input = self.tokenizer(
            [(prefix + input) if prefix is not None else '' for prefix, input in zip(prefixes, inputs)], postfixes,
            return_tensors="pt", truncation='only_first', padding=True
        )
        outputs = self.model.generate(
            tokenized_input.input_ids.to("cuda"),
            max_new_tokens=max_new_tokens,
        )
        return {
            'input': self.tokenizer.batch_decode(tokenized_input.input_ids),
            'output': self.tokenizer.batch_decode(outputs, skip_special_tokens=True),
        }


class T5(ModelInterface):
    def get_tokenizer(self, model_name):
        return T5Tokenizer.from_pretrained(model_name)

    def get_model(self, model_name):
        return T5ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16)


def get_model_interface(model_name):
    if model_name == 'google/flan-t5-xxl' or model_name == 'google/flan-t5-xl':
        return T5(model_name)
    else:
        raise NotImplementedError
