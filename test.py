import os
import sys
import pickle

# Load Conv AI local packages
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from conv_ai.test import TestSuite

# Data parsing
data_parsing = TestSuite("Data Parsing")

with open("./test/examples.pkl", "rb") as f:
    correct_examples = pickle.load(f)


def examples_is_list(examples):
    return type(examples) is list

data_parsing.add_test(examples_is_list, "examples is list")


def examples_format_check(examples):
    for example in examples:
        if "label" not in example:
            return False
        if "text" not in example:
            return False
    return True

data_parsing.add_test(examples_format_check, "examples each have text and label field")

def examples_check_length(examples):
    return len(correct_examples) == len(examples)


data_parsing.add_test(examples_check_length, "correct number of examples")


def examples_check_types(examples):
    return examples and type(examples[0]['text']) is str and type(examples[0]['label']) in [int, float]


data_parsing.add_test(examples_check_types, "example texts are strings and example labels are ints or floats")

def examples_check_labels(examples):
    return all([example['label'] in [-1, 1, -1.0, 1.0] for example in examples])


data_parsing.add_test(examples_check_labels, "example labels are either -1 or 1")
