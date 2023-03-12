from utils.constants import TRANSFORMER_XL, GPT2, GPT2_MEDIUM


def parse_string(s: str = None):
    """
    Parses a string representing a list of integers and ranges of integers and returns a list of integers.

    The input string should be in the format "a-b,c,d-e,...", where "a-b" represents a range of integers
    from a to b (inclusive), and "c" represents a single integer. The parts of the string should be separated by commas.

    This function can be used to parse strings representing lists of integers,
    for example when freezing specific layers or applying an accelerator to specific layers.

    :param s: The input string to parse.
    :return: A list of integers represented by the input string.
    """
    if not s:
        return []

    result = []
    for part in s.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def get_model_type(model_name: str = None):
    model_types = [GPT2_MEDIUM, GPT2, TRANSFORMER_XL]

    for model_type in model_types:
        if model_type in model_name:
            return model_type
