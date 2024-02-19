from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import torch


class ModelInterface:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = self.get_tokenizer(model_name)
        self.model = self.get_model(model_name)
        self.model.eval()
        self.default_generation_kwargs = {"max_new_tokens": 64}

    def to(self, device):
        self.model.to(device)

    def get_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def get_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16)

    def format_for_generation(self, prompts, only_truncate_prefix=None):
        return prompts, only_truncate_prefix

    def tokenize_and_generate(
            self, prompts, only_truncate_prefix=None, **generation_kwargs):
        gen_kwargs = {}
        gen_kwargs.update(self.default_generation_kwargs)
        gen_kwargs.update(generation_kwargs)
        if 'bad_words' in gen_kwargs.keys():
            assert 'bad_words_ids' not in gen_kwargs.keys()
            gen_kwargs['bad_words_ids'] = self.tokenizer(
                gen_kwargs['bad_words'], add_special_tokens=False).input_ids
            del gen_kwargs['bad_words']
        if only_truncate_prefix is None:
            tokenized_input = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True)
        else:
            assert isinstance(only_truncate_prefix, list)
            assert len(only_truncate_prefix) == len(prompts)
            prompts1 = [
                prompt[:i] for prompt, i in zip(prompts, only_truncate_prefix)]
            prompts2 = [
                prompt[i:] for prompt, i in zip(prompts, only_truncate_prefix)]
            tokenized_input = self.tokenizer(
                prompts1, prompts2, return_tensors="pt", padding=True,
                truncation='only_first')
        with torch.no_grad():
            device = next(iter(self.model.parameters())).device
            outputs = self.model.generate(
                tokenized_input.input_ids.to(device),
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )
        return tokenized_input.input_ids, outputs, gen_kwargs

    def get_output(self, input_ids, outputs, decoder_only=True):
        input_text = self.tokenizer.batch_decode(input_ids)
        output_text = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True)
        if decoder_only:
            input_text_wo_special_tokens = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True)
            output_text = [
                ot[len(it):] for it, ot in zip(
                    input_text_wo_special_tokens, output_text)]
        log_probs = torch.stack(outputs.scores, dim=1).log_softmax(-1)
        sequences = outputs.sequences[:, -log_probs.shape[1]:, None]
        gen_log_probs = torch.gather(
            log_probs, 2, sequences).squeeze(-1)
        confidence = torch.exp(gen_log_probs.mean(-1)).tolist()
        num_output_tokens = (gen_log_probs != 0).sum(-1)
        ends_early = (outputs.sequences[:, -1] != self.tokenizer.eos_token_id) \
            & (outputs.sequences[:, -1] != self.tokenizer.pad_token_id)
        return input_text, output_text, confidence, num_output_tokens, \
            ends_early

    def query(self, prompts, only_truncate_prefix=None, **generation_kwargs):
        prompts, only_truncate_prefix = self.format_for_generation(
            prompts, only_truncate_prefix=only_truncate_prefix)
        input_ids, outputs, full_gen_kwargs = self.tokenize_and_generate(
            prompts, only_truncate_prefix=only_truncate_prefix,
            **generation_kwargs)
        input_text, output_text, confidence, num_output_tokens, ends_early = \
            self.get_output(input_ids, outputs)
        return {
            'input': input_text,
            'output': output_text,
            'confidence': confidence,
            'num_output_tokens': num_output_tokens,
            'ends_early': ends_early,
            'generation_kwargs': full_gen_kwargs}


class T5(ModelInterface):
    def get_tokenizer(self, model_name):
        return T5Tokenizer.from_pretrained(model_name)

    def get_output(self, input_ids, outputs, decoder_only=True):
        return super().get_output(input_ids, outputs, decoder_only=False)

    def get_model(self, model_name):
        if torch.cuda.is_available():
            return T5ForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16)
        else:
            return T5ForConditionalGeneration.from_pretrained(
                model_name, device_map="auto")


INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{prompt}",
    response_key=RESPONSE_KEY,
)
class MPT(ModelInterface):
    def get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def get_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device, dtype=torch.bfloat16)
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_generation_kwargs = {
            # "temperature": 0.5,
            # "top_p": 0.92,
            # "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }

    def format_for_generation(self, prompts, only_truncate_prefix=None):
        prefix_length = PROMPT_FOR_GENERATION_FORMAT.index('{prompt}')
        new_only_truncate_prefix = None if only_truncate_prefix is None else \
            [prefix_length + x for x in only_truncate_prefix]
        new_prompts = []
        for prompt in new_prompts:
            new_prompts.append(
                PROMPT_FOR_GENERATION_FORMAT.format(prompt=prompt))
        return new_prompts, new_only_truncate_prefix


class Alpaca(MPT):
    def get_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained("/work/frink/mcinerney.de/stanford_alpaca/weights_7b")

    def get_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained("/work/frink/mcinerney.de/stanford_alpaca/weights_7b")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device, dtype=torch.bfloat16)
        return model


class Mistral(ModelInterface):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.default_generation_kwargs[
            "pad_token_id"] = self.tokenizer.pad_token_id
    def get_tokenizer(self, model_name):
        tokenizer = super().get_tokenizer(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.model_max_length = \
        #     AutoConfig.from_pretrained(model_name).max_position_embeddings
        tokenizer.model_max_length = 2000
        return tokenizer


class AlpaCare(ModelInterface):
    pass


def get_model_interface(model_name):
    if 'flan-t5' in model_name:
        return T5(model_name)
    elif 'mpt-' in model_name:
        return MPT(model_name)
    elif 'alpaca-' in model_name:
        return Alpaca(model_name)
    # elif model_name.startswith('mistralai/Mistral-'):
    elif 'mistralai' in model_name:
        return Mistral(model_name)
    elif 'AlpaCare-' in model_name:
        return AlpaCare(model_name)
    else:
        raise NotImplementedError
