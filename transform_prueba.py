
import re
import string
#import unidecode

import numpy as np
import pandas as pd

from collections import Counter

example_sentence = '2016-03-18 10:52:40'
test_case = ['hello', '...', 'h3.a', 'ds4,']

def remove_blanks(tokens):
    return [token.replace(" ", "") for token in tokens]

# tokens = remove_blanks(example_sentence)
# # print('Original:\n {}'.format(tokens))
# # print('\nRemoving blanks:\n {}'.format(tokens))
# print(tokens)
# print(example_sentence)

def remove_punctuation(sentence, keep_apostrophe=False):
    return re.sub(r'[^a-zA-Z0-9]', r' ', sentence)

def remove(array):
	for s in array:
		print(array)
		''.join(c for c in s if c not in string.punctuation) 

# tokens2 = remove_punctuation(tokens)
# print('Original:\n {}'.format(example_sentence))
# print('\nRemoving punctuation:\n {}'.format(tokens2))

test = remove(test_case)
# print(test_case)
print(test)

