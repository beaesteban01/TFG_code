
import re
import string
#import unidecode


import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from collections import Counter

example_sentence = "2016-03-18 10:52:40"  
test_case = ['hello', '...', 'h3.a', 'ds4,']

def remove_blanks(token):
	token.replace(" ", "")

# tokens = remove_blanks(example_sentence)
# # print('Original:\n {}'.format(tokens))
# # print('\nRemoving blanks:\n {}'.format(tokens))
# print(tokens)
# print(example_sentence)

def remove_punctuation(sentence, keep_apostrophe=False):

    return re.sub(r'[^a-zA-Z0-9]', r' ', sentence)
 


# Encode a numeric column as zscores
def encode_numeric_zscore(test2, mean=None, sd=None):
    if mean is None:
        mean = test2.mean()

    if sd is None:
        sd = test2.std()

    test2 = (test2 - mean) / sd

def remove(array):
	for s in array:
		print(array)
		''.join(c for c in s if c not in string.punctuation) 

# tokens2 = remove_punctuation(tokens)
# print('Original:\n {}'.format(example_sentence))
# print('\nRemoving punctuation:\n {}'.format(tokens2))

remove_punctuation(example_sentence)
print(example_sentence)
# test3_1 = test.replace(" ", "")
# test3_2 = int(test3_1)
# print(type(test3_1))
# print(type(test3_2))

#test3 = remove_blanks(test)
#test4 = encode_numeric_zscore(test3_2)
# print(test_case)
#print(test)
#print(test3_2)

