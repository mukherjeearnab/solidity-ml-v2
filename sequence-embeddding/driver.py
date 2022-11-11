from model import LSTM
from torch.autograd import Variable
import torch
import json
import pandas as pd


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)

# Import Dataset
url = "http://172.16.26.217:3000/5_Vectorization/CDS_vectorization_v1.csv"
data = pd.read_csv('./prepd-dataset.csv',
                   dtype={'opcode': object})

data['opcode'] = data['opcode'].apply(lambda x: json.loads(x))

data = data.iloc[:10]
print(data)


def pandas2tensor(data, X=True):
    tensor = Variable(torch.Tensor(data.tolist()))
    if X:
        return torch.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1]))
    else:
        return torch.reshape(tensor, (tensor.shape[0], 1))


OPCODE_SEQ_LEN = 1800
EMBEDDING_DIM = 50
VOCAB_SIZE = 150
NUM_EPOCHS = 128
# BATCH_SIZE = 128

LEARNING_RATE = 0.001  # 0.001 lr
# INPUT_SIZE = 1800  # number of features
HIDDEN_SIZE = 2  # number of features in hidden state
NUM_LAYERS = 1  # number of stacked lstm layers
NUM_CLASSES = 1  # number of output classes

train_sequences = pandas2tensor(data['opcode'])
train_labels = pandas2tensor(data['swc_label'], X=False)

train_sequences.to(device)
train_labels.to(device)

# Define the Neural Network Structure (Layers)
model = LSTM(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, embd_dim=EMBEDDING_DIM,
             hidden_size=HIDDEN_SIZE, seq_length=train_sequences.shape[1], num_layers=NUM_LAYERS)

model_state = model.export()


print(train_labels)


print("Train-Sequences", train_sequences.shape, type(train_sequences[0]))
print("Train-Labels", train_labels.shape, type(train_labels[0]))


for index, (sequence, label) in enumerate(zip(train_sequences, train_labels)):
    print(f'Fitting {index} of {train_sequences.shape[0]}')

    model.load(model_state)

    # print(model.model)

    # Compile the model
    model.compile(learning_rate=LEARNING_RATE)

    # Fit and train the RNN model with Training and Validiation Data
    model.fit(num_epochs=NUM_EPOCHS, X=sequence, y=label)

    output = model.embeddings(sequence, label)
    with open(f'./embeddings/{index}.txt', 'w') as f:
        f.write(str(output[0].tolist()))

    del model
