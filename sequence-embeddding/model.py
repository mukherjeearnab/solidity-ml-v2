import torch
import torch  # pytorch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)

# Set Torch seet to preserve deterministic model initiation
torch.manual_seed(0)


class LSTMNet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embd_dim: int,
                 hidden_size: int, seq_length: int, num_layers=1):
        super(LSTMNet, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers    # number of layers
        self.embd_dim = embd_dim        # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length    # sequence length

        # Embedding Layer
        self.embeddings = nn.Embedding(
            vocab_size, embd_dim, padding_idx=0)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embd_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # Fully Connected 1
        self.fc_1 = nn.Linear(hidden_size, 64)

        # Output Layer
        self.fc = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()        # ReLU Activation Function
        self.sigmoid = nn.Sigmoid()  # Sigmoid Activation Function

    def forward(self, x: torch.Tensor):

        # Embedd the input sequence
        x = self.embeddings(x.int())

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # LSTM hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # LSTM internal state

        # Propagate input through LSTM
        # LSTM with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Reshaping the data for Dense layer next
        hn = hn.view(-1, self.hidden_size)

        out = self.relu(hn)     # ReLU
        out = self.fc_1(out)    # first Dense

        # Return First Dense's output if in eval mode (THE OBJECTIVE)
        if not self.training:
            # print("DenseInt", out)
            return out

        out = self.relu(out)        # ReLU
        out = self.fc(out)          # Final Output
        out = self.sigmoid(out)     # Sigmoid
        return out


class LSTM():
    def __init__(self, vocab_size: int, num_classes: int, embd_dim: int,
                 hidden_size: int, seq_length: int, num_layers=1) -> LSTMNet:

        # Init LSTM Model with provided params
        self.model = LSTMNet(vocab_size, num_classes, embd_dim,
                             hidden_size, seq_length, num_layers)

        # Convert model to device, i.e. if CUDA available
        self.model.to(device)

    def compile(self, learning_rate: float):
        '''
        Method to compile model and init Loss Function and Optimizer
        with Learning Rate
        '''
        self.criterion = torch.nn.BCELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def fit(self, num_epochs: int, X: torch.Tensor, y: torch.Tensor):
        '''
        Method to fit the data into the model
        '''

        # Convert input tensors to CUDA tensors, if available
        X.to(device)
        y.to(device)

        sequence, label = X, y

        # for num_epochs, train the model
        for epoch in range(num_epochs):

            # init accuracy to 0
            accuracy = 0.0

            # Fit individual sequence to the model
            # for (sequence, label) in zip(X, y):
            outputs = self.model.forward(sequence)  # forward pass
            # caluclate the gradient, manually setting to 0
            self.optimizer.zero_grad()

            # obtain the loss function
            loss = self.criterion(outputs[0], label)

            loss.backward()         # calculate the loss from the loss function
            self.optimizer.step()   # improve from loss, i.e backprop

            # calculate train accuracy
            output = (outputs >= 0.85).float()
            accuracy += (output == label)

            # calculate average accuracy of total set of data
            accuracy = accuracy/y.shape[0]
            accuracy = accuracy.detach().item()
            loss = loss.item()

            # print per epoch stats
            print(f'Epoch {epoch} of {num_epochs}. Loss={loss} Acc={accuracy}',
                  end='\r')

            if int(accuracy) == 1 and loss < 0.15:
                print('\nTraining Complete!')
                break

    def __evaluate(self, X: torch.Tensor, y: torch.Tensor, use_cuda=False):
        self.model.eval()
        with torch.no_grad():
            # init accuracy
            acc = .0

            # convert to CUDA if available
            X.to(device)
            y.to(device)

            # predict the outcome
            y_preds = self.model(X)

            # convert prob into binary class
            y_preds = (y_preds > 0.5).float()

            # calc accuracy
            acc = (y_preds == y).sum()/float(y_preds.shape[0])
        return acc.detach().item()

    def embeddings(self, X: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            # convert to CUDA if available
            X.to(device)
            y.to(device)

            # predict the outcome
            embeddings = self.model(X)

        return embeddings.detach()
