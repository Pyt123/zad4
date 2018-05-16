import torch
from torch import nn
import pickle as pkl
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 64
HIDDEN_SIZES = [256, 256]

NUM_OF_CLASSES = 36
INPUT_RESOLUTION = 56


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_RESOLUTION * INPUT_RESOLUTION, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        #self.fc3 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.fc4 = nn.Linear(HIDDEN_SIZES[1], NUM_OF_CLASSES)

    def forward(self, x):
        x = x.view(-1, INPUT_RESOLUTION * INPUT_RESOLUTION)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return self.fc4(x)
        #x = (self.fc1(x))
        #x = (self.fc2(x))
        #x = (self.fc3(x))
        #return (self.fc4(x))



def train():
    COUNT = 27500
    EPOCHS = 3000
    START_MOMENTUM = 0.03
    MOMENTUM = START_MOMENTUM
    DIVIDER = 1.5
    EPOCHS_TO_CHANGE = 500
    NEXT_TO_CHANGE = EPOCHS_TO_CHANGE

    # Load data
    (x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
    (x_train, y_train) = (x_train[:COUNT], y_train[:COUNT])

    # Create model
    #model = NeuralNet()()
    # Load model
    model = torch.load('mytraining.pth')

    # Some stuff
    optimizer = optim.SGD(model.parameters(), lr=0.006, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    model.train()

    # Convert numpy arrays to torch variables
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.FloatTensor), requires_grad=True)
    targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)

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

        if (epoch) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.24f}'.format(epoch, EPOCHS, loss.data[0]))

    torch.save(model, 'mytraining.pth')
    return model
