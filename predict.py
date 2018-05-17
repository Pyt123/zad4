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
        nump = parameter.cpu().type(torch.FloatTensor).data.numpy().astype(np.float16)
        np.save('model/params' + str(i), nump)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'), np.load('model/params3.npy'), np.load('model/params4.npy'))


def save_model_as_txt(par1, par2, par3, par4):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)


def relu(x):
    return x * (x > 0)


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    (w1, p1, w2, p2) = load_model_from_file()
    i = 0
    wyniki_kurwa = []
    for x_n in x:
        print('1 ' + str(x_n.shape) + '\trazy: ' + str(w1.shape))
        layer1 = relu(np.dot(x_n, np.transpose(w1)))

        #print('2 ' + str(layer1.shape) + '\trazy: ' + str(p1.shape))
        #layer2 = sigmoid(np.dot(layer1, p1))

        print('3 ' + str(layer1.shape) + '\trazy: ' + str(w2.shape))
        layer3 = relu(np.dot(layer1, np.transpose(w2)))

        #print('4 ' + str(layer3.shape) + '\trazy: ' + str(p2.shape))
        #do_app = (sigmoid(np.dot(layer3, p2)))
        do_app = layer3
        arg = do_app.argmax()
        print('końcowa: ' + str(do_app.shape))
        wyniki_kurwa.append(arg)
        i += 1
        print(str(i))
    output = np.array(wyniki_kurwa)
    np.save('predicted_vals', output)

    good = 0
    for i in range(0, 2500):
        if output[i] == y_train[i]:
            good += 1
    print("good: " + str(good))
    print("ratio: " + str(good/2500.0 * 100))
    return output


#content.train()

(x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
(x_train, y_train) = (x_train[27500:], y_train[27500:])
#targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)
#print(predicted_y)
model = torch.load('mytraining.pth', 'cpu')
save_model_as_numpy(model)
#save_model_as_txt(par1, par2, par3, par4, par5, par6)
predicted_y = predict(x_train)

#print('1:\t' + str(len(par1)) + ' x ' + str(len(par1.transpose())))
#print('2:\t' + str(len(par2)) + ' x ' + str(len(par2.transpose())))
#print('3:\t' + str(len(par3)) + ' x ' + str(len(par3.transpose())))
#print('4:\t' + str(len(par4)) + ' x ' + str(len(par4.transpose())))
#print('5:\t' + str(len(par5)) + ' x ' + str(len(par5.transpose())))
#print('6:\t' + str(len(par6)) + ' x ' + str(len(par6.transpose())))
#pred = np.load('predicted_vals.npy')
np.savetxt('predicted_vals.txt', pred, fmt='%d')
np.savetxt('y_train.txt', y_train, fmt='%d')
exit(0)
print('')


