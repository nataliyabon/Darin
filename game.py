import torch
from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

%matplotlib inline


class CNN(torch.nn.Module):    
    def __init__(self):
        super(CNN, self).__init__()
        #2 * 16 * 16
        self.conv1 = torch.nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # 16 * 16 * 16
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 16 x 8 x 8
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 32 x 8 x 8
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 64 * 8 * 8
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # 128 * 8 * 8
        #pool
        # 128 x 4 x 4
        self.conv6 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # 256 x 4 x 4
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = torch.nn.Linear(1024, 255)
        
    def forward(self, x):
        x = x.view(-1, 2, 16, 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)


def model_policy(clicked, step, model, not_used):
    if step % 2 == 0:
        walk = 1
    else:
        walk = -1
    #i, j = gm_has_four_line(walk, clicked)
    #if i != -1 and j != -1:
     #   print(i + 1, j + 1, 'FIND')
      #  return i * 15 + j
    
    #сделаем тензор
    data = np.zeros((2, 16, 16))
    for i in range(15):
        for j in range(15):
            data[0][i][j] = clicked[i][j]
    data[1] = np.ones((16, 16)) * walk
    data = torch.from_numpy(data)
    data = data.float()
    ans = model(data)

    max_ind = torch.max(ans, 1)
    max_ind_1 = max_ind[1].numpy()
    
    while((max_ind_1[0] + 1) not in not_used):
        ans[0][max_ind_1[0]] = ans[0][max_ind_1[0]] - 100
        max_ind = torch.max(ans, 1)
        max_ind_1 = max_ind[1].numpy()
    
    return max_ind_1[0]


def string_to_clicked(s):
    step = 0
    clicked = np.zeros((15, 15), dtype = np.int8)
    game = s.split()
    gamer = 1

    clicked = np.zeros((15, 15), dtype = np.int8)  # Создаем сет для клеточек, по которым мы кликнули  
    not_used = list(range(1, 15*15 + 1))

    for move in game:
        if move[0] > 'p' or move[0] < 'a':
            break
            
        cell = -1

        if (step % 2 == 0):
            cell = 1
        x = ord(move[0]) - ord('a') - 1
        if (move[0] > 'i'):
            x -= 1
        y = int(move[1:]) - 1
        clicked[x][y] = cell
        step += 1
        
    if step % 2 == 0:
        step = 1
    else:
        step = -1


    return step, clicked, not_used


def gm_has_four_line(gamer, clicked):
    w = 0
    for i in range(15):     #по строкам
        w = 0
        for j in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 4:
                if j < 14 and clicked[i][j + 1] == 0:
                    return i, j + 1
                elif j - 4 >= 0 and clicked[i][j - 4] == 0:
                    return i, j - 4
            
    for j in range(15):          #по столбцам
        w = 0
        for i in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 4:
                if i < 14 and clicked[i + 1][j] == 0:
                    return i + 1, j
                elif i - 4 >= 0 and clicked[i - 4][j] == 0:
                    return i - 4, j
           
    for i in range(12):
        w = 0                                      #по диагонали нижняя половина
        for k in range(15 - i):
            if clicked[i + k][k] == gamer:
                w += 1
            else:
                w = 0

            if w >= 4:
                if i + k + 1 <= 14 and k + 1 <= 14 and clicked[i + 1 + k][k + 1] == 0:
                    return i + 1 + k, k + 1
                elif i + k - 4 >= 0 and k - 4 >= 0 and clicked[i + k - 4][k - 4] == 0:
                    return i + k - 4, k - 4
       
    for i in range(12):
        w = 0                                      #по диагонали верхная половина
        for k in range(15 - i):
            if clicked[k][i + k] == gamer:
                w += 1
            else:
                w = 0
            if w >= 4:
                if i + k + 1 <= 14 and k + 1 <= 14 and clicked[1 + k][k + i + 1] == 0:
                    return 1 + k, k + 1 + i
                elif i + k - 4 >= 0 and k - 4 >= 0 and clicked[k - 4][k + i - 4] == 0:
                    return k - 4, k - 4 + i
                       
            
    return -1, -1

def to_out(cell):
    x = cell % 15
    y = cell // 15
    out = ord('a') + x
    ans = chr(out) + str(y) + '\n'
    return ans


def run_model(model, log):
    clicked = np.zeros((15, 15), dtype = np.int8)  # Создаем сет для клеточек, по которым мы кликнули
    not_used = list(range(1, 15*15 + 1))
    step, clicked, not_used = string_to_clicked(log)
    not_used = list(range(1, 15*15 + 1))

    for i in range(1, 15*15 + 1):
        if clicked[(i - 1) % 15][(i - 1) // 15] != 0:
            not_used.remove(i)

    cell = model_policy(clicked, step, model, not_used)

    return(to_out(cell))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = CNN()
    model.load_state_dict(torch.load("/home/nata/model_rand_3", map_location='cpu'))
    model.eval()

    log = sys.stdin.readline().decode()

    ans = run_model(model, log)

    sys.stdout.write(ans.encode())
    sys.stdout.flush()
