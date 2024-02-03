"""
parsers that combine outputs from multiple chunks.
"""
import re
from typing import List, Tuple, Union, Optional


class BaseParser:
    def parse_output(self, output: str) -> Union[str, int]:
        raise NotImplementedError

    def combine_results(self, results: List[Union[str, int]]) \
            -> Union[str, int]:
        raise NotImplementedError

    def __call__(self, outputs: List[str]) \
            -> Tuple[List[Union[str, int]], Union[str, int]]:
        results = [
            self.parse_output(output) for output in outputs]
        final_result = self.combine_results(results)
        return results, final_result


class ClassificationParser(BaseParser):
    def __init__(self, class_info: List[Tuple[Optional[str], Union[str, int]]],
                 class_priority: List[Union[str, int]], full_match: bool = True):
        """
        class_info: a list of tuples of the form
                (<regular expression>, <label>).
            The regular expressions will be checked in order and when a
            match is reached, the corresponding label will be predicted.
            A regular expression of None should be given for the last
            class to catch all previously uncaught outputs. all other
            regular expressions should be strings.
        
        class_priority: the labels listed in the order of priority: the
            highest priority labels (at the top of the list) will
            superseed lower priority ones during aggregation.
        """
        self.class_info = class_info
        self.full_match = full_match
        assert len(self.class_info) > 1
        assert self.class_info[-1][0] is None
        self.class_priority = class_priority
        self.class_priority_map = {
            label: i for i, label in enumerate(class_priority)}

    def parse_output(self, output: str) -> Union[str, int]:
        for regex, label in self.class_info[:-1]:
            assert isinstance(regex, str)
            if self.full_match:
                if re.fullmatch(regex, output, flags=re.DOTALL) is not None:
                    return label
            else:
                if re.search(regex, output) is not None:
                    return label
        return self.class_info[-1][1]

    def combine_results(self, results: List[Union[str, int]]) \
            -> Union[str, int]:
        priority_indices = [
            self.class_priority_map[result] for result in results]
        priority_idx = min(priority_indices)
        return self.class_priority[priority_idx]
