import re
import yaml
from yaml.scanner import ScannerError
from .parsers import ClassificationParser


def process_string_output(output, needs_letters=True):
    # strip, lowercase, delete all non-alphanumeric characters, and standardize all whitespaces
    new_output = ' '.join(re.sub('[\W_]+', ' ', output.strip().lower()).split())
    if needs_letters:
        new_output = '' if re.search('[a-zA-Z]', new_output) is None else new_output
    return new_output


yes_no_parser = ClassificationParser(
    class_info=[("(yes|Yes|YES)([\s,.]+.*)?", 1), (None, 0)],
    class_priority=[1, 0])
def process_yes_no_output(output):
    return yes_no_parser([process_string_output(output)])[1] == 1


def process_set_output(output):
    if output.strip().lower().startswith('no ') or \
            output.strip().lower().startswith('none ') or \
            output.strip().lower().startswith('1. none ') or \
            output.strip().lower().startswith('* none '):
        return set()
    # decide what to consider a delimiter
    if output.strip().startswith('1. ') or '\n1. ' in output.strip():
        output = set(re.split('\n[0-9]+\. ', output[3:])[1:])
    else:
        delimiter = '\n* ' if '\n* ' in output else \
            ' -' if ' -' in output else ';' if ';' in output else ','
        output = ' ' + output
        # split string by delimiter
        output = set([process_string_output(x) for x in output.split(delimiter)])
    # delete empty strings and substrings of those that already appear in the set
    output = set([x for x in output if x != '' and all([not x in y for y in output - {x}])])
    # if none is the only element, return the empty set
    if len(output) == 1 and list(output)[0] == 'none':
        return set()
    return output


def truncate_if_ends_early(outputs):
    output = outputs['output']
    if outputs['ends_early']:
        output = '\n'.join(output.split('\n')[:-1])
    return output


def yaml_postprocess(output):
    if "```yaml\n" in output:
        output_yaml = output.split("```yaml\n")[1].split("\n```")[0]
    else:
        # if no yaml start indicator exists,
        # assume everything is part of the yaml
        output_yaml = output
    not_successful = True
    i = 0
    split_output_yaml = output_yaml.split('\n')
    while not_successful:
        try:
            output_yaml_temp = '\n'.join(split_output_yaml[:-i]) \
                if i > 0 else output_yaml
            output = yaml.safe_load(output_yaml_temp)
            not_successful = False
        except Exception as e:
            i += 1
            if i >= len(split_output_yaml) or i > 10: # maximum 10 tries
                break
    if not_successful:
        return None
    return output
