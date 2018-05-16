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

def save_model_as_numpy(model):
    i = 1
    for parameter in model.parameters():
        nump = parameter.cpu().data.numpy()
        np.save('model/params' + str(i), nump)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'), np.load('model/params3.npy'),
            np.load('model/params4.npy'), np.load('model/params5.npy'), np.load('model/params6.npy'))


def save_model_as_txt(par1, par2, par3, par4, par5, par6):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)
        np.savetxt('model/params5.txt', par5)
        np.savetxt('model/params6.txt', par6)


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    model = torch.load('mytraining.pth')
    model.eval()

    x = torch.autograd.Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad=True)

    #for parameter in model.parameters():
     #   print(parameter)

    good = 0
    predicted = np.array([], dtype=np.int)
    for i in range(0, 2500):
        output = model(x[i])
        _, pred = output.data.max(-1, keepdim=True)
        if pred[0, 0] == int(y_train[i]):
            good += 1
    print("good: " + str(good))
    print("ratio: " + str(good/2500.0 * 100))
    return predicted


#content.train()

#(x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
#(x_train, y_train) = (x_train[27500:], y_train[27500:])
#targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)
#predicted_y = predict(x_train)
#print(predicted_y)
model = torch.load('mytraining.pth')
#save_model_as_numpy(model)
(par1, par2, par3, par4, par5, par6) = load_model_from_file()
save_model_as_txt(par1, par2, par3, par4, par5, par6)
print('')


