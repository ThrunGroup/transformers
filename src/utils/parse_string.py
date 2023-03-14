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
    if not s or s == "None":
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


def string_to_dict(s: str) -> dict:
    """
    Convert a string of accelerator arguments into a dictionary.

    This function can be used to extract accelerator arguments information from the model name.

    Example:
    If the string is "a:10,k:20", this function will return a dictionary of {'a': 10, 'k': 20}.

    Args:
        s (str): The string of accelerator arguments to convert to a dictionary.

    Returns:
        dict: The dictionary form of the string.
    """
    if not s:
        return {}
    pairs = s.split(",")
    pairs = [pair.split(":") for pair in pairs]
    dic = {k: int(v) for (k, v) in pairs}
    return dic


def dict_to_string(dic: dict) -> str:
    """
    Convert a dictionary of accelerator arguments into a string.

    This function can be used to name the model in the "create_model" function inside create_models.py.

    Example:
    If the accelerator arguments are {'a': 10, 'k': 20}, the output string should be "a:10,k:20".

    :param: dic (dict): The dictionary of accelerator arguments to convert to a string.
    :return: The string form of the dictionary.
    """
    if dic == None:
        return "None"
    return ",".join([f"{k}:{v}" for (k, v) in dic.items()])
