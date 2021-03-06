import torch
from torch import nn
import pickle as pkl
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 64
HIDDEN_SIZES = [768]

NUM_OF_CLASSES = 36
INPUT_RESOLUTION = 56


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_RESOLUTION * INPUT_RESOLUTION, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], NUM_OF_CLASSES)

    def forward(self, x):
        x = x.view(-1, INPUT_RESOLUTION * INPUT_RESOLUTION)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train():
    COUNT = 27500
    EPOCHS = 5000
    START_MOMENTUM = 0.1
    MOMENTUM = START_MOMENTUM
    DIVIDER = 1.2
    EPOCHS_TO_CHANGE = 100
    NEXT_TO_CHANGE = EPOCHS_TO_CHANGE

    # Load data
    (x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
    (x_train, y_train) = (x_train[:COUNT], y_train[:COUNT])

    # Create model
    #model = NeuralNet().cuda()
    # Load model
    model = torch.load('mytraining.pth')

    # Some stuff
    optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-3)
    #optimizer = optim.SGD
    criterion = nn.CrossEntropyLoss()

    model.train()

    # Convert numpy arrays to torch variables
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
    targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)

    for epoch in range(EPOCHS):
        if epoch == NEXT_TO_CHANGE:
            MOMENTUM /= DIVIDER
            NEXT_TO_CHANGE += EPOCHS_TO_CHANGE
            torch.save(model, 'mytraining.pth')

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch [{}/{}],\tLoss: {:.24f}'.format(epoch, EPOCHS, loss.data[0]))

    torch.save(model, 'mytraining.pth')
    return model



'''def predict(x):
    x_train, y_train = pkl.load(open('train.pkl', mode='rb'))
    x_train, y_train = x_train[:6], y_train[:6]
    return y_train[np.argmin(
        2.5 * x.astype(np.uint16) @ ~x_train.astype(np.bool).transpose() + ~x.astype(np.bool) @ x_train.astype(
            np.uint16).transpose(), axis=1)]


x_train, y_train = pkl.load(open('train.pkl', mode='rb'))
x_train = x_train[6:10]
pred = predict(x_train)
print(type(pred))
print(pred.shape)'''
