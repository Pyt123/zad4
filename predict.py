# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import content
import torch
import pickle as pkl
import numpy as np

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    model = torch.load('mytraining.pth')
    model.eval()

    x = torch.autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True)
    good = 0
    predicted = np.array([], dtype=np.int)
    for i in range(0, 2500):
        output = model(x[i])
        #if i == 0:
            #print(pred)
            #print(int(y_train[i]))
        _, pred = output.data.max(-1, keepdim=True)
        #if i < 20:
            #.data.cpu().numpy()[0]
            #print(pred[0, 0])
            #print(int(y_train[i]))
        if pred[0, 0] == int(y_train[i]):
            good += 1
        #pred_val = pred.cpu().numpy()[0][0]
        #print(pred)
        #predicted = np.append(predicted, pred_val, axis=0)
    print("good: " + str(good))
    print("ratio: " + str(good/2500.0 * 100))
    return predicted


content.train()

(x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
(x_train, y_train) = (x_train[27500:], y_train[27500:])
targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)
predicted_y = predict(x_train)
#print(predicted_y)