import numpy as np
import random
from tkinter import *
import random
import torch
import torch.nn.functional as F

clicked = np.zeros((15, 15), dtype = np.int8)  # Создаем сет для клеточек, по которым мы кликнули
not_used = list(range(1, 16*16 + 1))

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

def kclick(event):
    print("black or white? (b/w)")
    ans = input()
    if (ans == 'w'):
        white_click(event)

def from_15_to_16(num):
    return num + (num // 15)


def from_16_to_15(num):
    return num - (num // 16)

def click(event):
    print(not_used)

    ids = c.find_withtag(CURRENT)[0]  # Определяем по какой клетке кликнули
    x = (ids - 1) % 15
    y = (ids - 1) // 15
    cell = from_15_to_16(ids)
    clicked[x][y] = 1
    not_used.remove(cell)
    print(cell)
    c.itemconfig(CURRENT, fill="#C7007D")
        
    cell = model_policy(clicked, 1, model, not_used) + 1
    ids = from_16_to_15(cell)
    not_used.remove(cell)
    print(cell)

    clicked[(ids - 1) % 15][(ids - 1) // 15] = -1
    c.itemconfig(ids, fill="#1CA9C9")
    #print(ids, (ids - 1) % 15, (ids - 1) // 15)
    c.update()
    game_is_over = game_over()
    if (game_is_over != 0):
        print("GAME OVER, WINNER IS ", game_is_over)

def model_policy(clicked, step, model, not_used):
    if step % 2 == 0:
        walk = 1
    else:
        walk = -1
    i, j = gm_has_four_line(walk, clicked)
    if i != -1 and j != -1:
        return j * 16 + i

    i, j = gm_has_three_line(walk, clicked)
    if i != -1 and j != -1:
        return j * 16 + i

    i, j = gm_has_two_line(walk, clicked)
    if i != -1 and j != -1:
        return j * 16 + i
    
    
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
    
# || max_ind_1[0] % 16 == 15 || max_ind_1[0] // 16

    while((max_ind_1[0] + 1) not in not_used or (max_ind_1[0] + 1) % 16 == 15 or max_ind_1[0] >= 16*15):
        ans[0][max_ind_1[0]] = ans[0][max_ind_1[0]] - 100
        max_ind = torch.max(ans, 1)
        max_ind_1 = max_ind[1].numpy()
    
    return max_ind_1[0]

def to_pos(move):
    x = ord(move[0]) - ord('a')
    if (move[0] > 'i'):
            x -= 1
    y = int(move[1:]) - 1
    return x, y


def string_to_clicked(s):
    step = 0
    clicked = np.zeros((16, 16), dtype = np.int8)
    game = s.split()
        
    for move in game:
        if move[0] > 'p' or move[0] < 'a':
            break
            
        cell = -1
        if (step % 2 == 0):
                cell = 1
        x, y = to_pos(move)
        clicked[x][y] = cell
        step += 1
        
    cell = -1
    if (step % 2 == 0):
            cell = 1

    return step, clicked


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


def gm_has_three_line(gamer, clicked):
    w = 0
    for i in range(15):     #по строкам
        w = 0
        for j in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 3:
                if j < 14 and clicked[i][j + 1] == 0 and j - 3 >= 0 and clicked[i][j - 3] == 0:
                    return i, j + 1

            
    for j in range(15):          #по столбцам
        w = 0
        for i in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 3:
                if i < 14 and i - 3 >= 0 and clicked[i + 1][j] == 0 and clicked[i - 3][j] == 0:
                    return i + 1, j
           
    for i in range(13):
        w = 0                                      #по диагонали нижняя половина
        for k in range(15 - i):
            if clicked[i + k][k] == gamer:
                w += 1
            else:
                w = 0

            if w >= 3:
                if i + k + 1 <= 14 and k + 1 <= 14 and i + k - 3 >= 0 and k - 3 >= 0:
                    if clicked[i + 1 + k][k + 1] == 0 and clicked[i + k - 3][k - 3] == 0:
                        return i + 1 + k, k + 1

       
    for i in range(13):
        w = 0                                      #по диагонали верхная половина
        for k in range(15 - i):
            if clicked[k][i + k] == gamer:
                w += 1
            else:
                w = 0
            if w >= 3:
                if i + k + 1 <= 14 and k + 1 <= 14 and i + k - 3 >= 0 and k - 3 >= 0:
                    if clicked[1 + k][k + i + 1] == 0 and clicked[k - 3][k + i - 3] == 0:
                        return 1 + k, k + 1 + i
                       
            
    return -1, -1


def gm_has_two_line(gamer, clicked):
    w = 0
    for i in range(15):     #по строкам
        w = 0
        for j in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 2:
                if j < 14 and clicked[i][j + 1] == 0 and j - 2 >= 0 and clicked[i][j - 2] == 0:
                    return i, j + 1

            
    for j in range(15):          #по столбцам
        w = 0
        for i in range(15):
            if clicked[i][j] == gamer:
                w += 1
            else:
                w = 0
            if w >= 2:
                if i < 14 and i - 2 >= 0 and clicked[i + 1][j] == 0 and clicked[i - 2][j] == 0:
                    return i + 1, j
           
    for i in range(13):
        w = 0                                      #по диагонали нижняя половина
        for k in range(15 - i):
            if clicked[i + k][k] == gamer:
                w += 1
            else:
                w = 0

            if w >= 2:
                if i + k + 1 <= 14 and k + 1 <= 14 and i + k - 2 >= 0 and k - 2 >= 0:
                    if clicked[i + 1 + k][k + 1] == 0 and clicked[i + k - 2][k - 2] == 0:
                        return i + 1 + k, k + 1

       
    for i in range(13):
        w = 0                                      #по диагонали верхная половина
        for k in range(15 - i):
            if clicked[k][i + k] == gamer:
                w += 1
            else:
                w = 0
            if w >= 2:
                if i + k + 1 <= 14 and k + 1 <= 14 and i + k - 2 >= 0 and k - 2 >= 0:
                    if clicked[1 + k][k + i + 1] == 0 and clicked[k - 2][k + i - 2] == 0:
                        return 1 + k, k + 1 + i
                       
            
    return -1, -1

def to_out(cell):
    x = cell % 16
    y = cell // 16 + 1
    out = ord('a') + x
    if (out >= ord('i')):
        out += 1
    ans = chr(out) + str(y)
    return ans

def make_not_used(clicked):
    not_used = list(range(1, 256))
    for i in range(1, 16*16 + 1):
        if clicked[(i - 1) % 16][(i - 1) // 16] != 0:
            not_used.remove(i)

    return not_used

def run_model(model, log):

    clicked = np.zeros((15, 15), dtype = np.int8)  # Создаем сет для клеточек, по которым мы кликнули
    step, clicked = string_to_clicked(log)
    not_used = make_not_used(clicked)


    cell = model_policy(clicked, step, model, not_used)

    return(to_out(cell))

def game_over():
    b = 0 
    w = 0
    for i in range(15):     #по строкам
        w = 0
        b = 0
        for j in range(15):
            if (clicked[i][j]) == 1:
                b += 1
                w = 0
            elif clicked[i][j] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1
            
    for j in range(15):          #по столбцам
        w = 0
        b = 0
        for i in range(15):
            if (clicked[i][j]) == 1:
                b += 1
                w = 0
            elif clicked[i][j] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1
           
    for i in range(11):
        w = 0                                      #по диагонали нижняя половина
        b = 0
        for k in range(15 - i):
            if (clicked[i + k][k]) == 1:
                b += 1
                w = 0
            elif clicked[i + k][k] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0

            if b >= 5:
                return 1
            if w >= 5:
                return -1
       
    for i in range(11):
        w = 0                                      #по диагонали верхная половина
        b = 0
        for k in range(15 - i):
            if (clicked[k][i + k]) == 1:
                b += 1
                w = 0
            elif clicked[k][i + k] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1
                       
            
    return 0



model = CNN()
model.load_state_dict(torch.load('model_rand_16_10', map_location='cpu'))
model.eval()

GRID_SIZE = 15 # Ширина и высота игрового поля
SQUARE_SIZE = 30 # Размер одной клетки на поле

global step
step = 0
    
root = Tk() # Основное окно программы
root.title("連珠")
c = Canvas(root, width=GRID_SIZE * SQUARE_SIZE, height=GRID_SIZE * SQUARE_SIZE) # Задаем область на которой будем рисовать
c.pack()
 
c.bind("<Button-1>", click)
# Следующий код отрисует решетку из клеточек серого цвета на игровом поле
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        c.create_rectangle(i * SQUARE_SIZE, j * SQUARE_SIZE,
                           i * SQUARE_SIZE + SQUARE_SIZE,
                           j * SQUARE_SIZE + SQUARE_SIZE, fill='#FFCBDB')
 
root.mainloop() # Запускаем программу