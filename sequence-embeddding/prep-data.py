from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import pandas as pd

# read the dataset
data = pd.read_csv('http://172.16.26.217:3000/merge.csv',
                   dtype={'swc_label': object, 'opcode': object})

# remove corrupted data
for index, row in data.iterrows():
    if not isinstance(row['opcode'], str):
        data = data.drop([index])

data = data.iloc[:100]

# method to remove operands from opcode sequence


def re_hex_val(opcode):
    opcode = str(opcode)
    opcode = opcode.replace('|', '')
    regex = '(0x|0X)[a-fA-F0-9]+ '
    import re
    opcode = re.sub(regex, '', opcode)
    opcode = opcode.strip()
    return opcode


# remove operands
data['opcode'] = data['opcode'].apply(re_hex_val)


# define tokenizing params
OPCODE_SIZE = 150
OPCODE_SEQ_LEN = 1800
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = '<OOV>'

# execute tokenizer
tokenizer = Tokenizer(num_words=OPCODE_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(data['opcode'])
word_index = tokenizer.word_index

# print words, i.e., opcodes
print(word_index.items())

# Save Tokenizer as Pickle
with open('tokenizer.pickle', 'wb') as fh:
    pickle.dump(tokenizer, fh)

# Tokenize OPCODES
tokenized_opcodes = tokenizer.texts_to_sequences(data['opcode'])

# Pad TOkenized OPCODES
padded_opcodes = pad_sequences(
    tokenized_opcodes, maxlen=OPCODE_SEQ_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

opcodes = np.array(padded_opcodes).tolist()

data['opcode'] = opcodes

# convert to binary labels
data['swc_label'] = (data['swc_label'].str.contains('1')).astype(int)

# filter dataset to final dataset.
final_data = data[['opcode', 'swc_label']]


final_data.to_csv('./prepd-dataset.csv', index=False)
