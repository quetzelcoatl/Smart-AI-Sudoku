import copy

import cv2
from matplotlib import pyplot as plt, font_manager, rcParams

N = 9

def correction(arr):
    for row in range(9):
        temp = []
        for col in range(9):
            if arr[row][col] not in temp:
                temp.append(arr[row][col])
            else:
                grid[row][col] = 0

    for col in range(9):
        temp = []
        for row in range(9):
            if arr[row][col] not in temp:
                temp.append(arr[row][col])
            else:
                grid[row][col] = 0



def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end=" ")
        print()


def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3

    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


def solveSuduko(grid, row, col):
    if row == N - 1 and col == N:
        return True

    if col == N:
        row += 1
        col = 0

    if grid[row][col] > 0:
        return solveSuduko(grid, row, col + 1)

    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num

            if solveSuduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


grid = []
with open("grid.txt", "r") as text_file:
    temp = []
    for i in range(0, 9):
        line = text_file.readline()
        for ele in line[:-1]:
            temp.append(int(ele))
        grid.append(temp)
        temp = []
_grid = copy.deepcopy(grid)
correction(grid)
#print(grid)
if (solveSuduko(grid, 0, 0)):
    printing(grid)
else:
    print("Reupload Image")

image_orig = cv2.imread('output_grid.PNG')
image = copy.copy(image_orig)
font = cv2.FONT_HERSHEY_SIMPLEX

color_green = (0, 107, 56) #006B38FF
color_black = (16,24,32) #101820FF
x = 60
y = 60
org = (60, 60)
fontScale = 1
thickness = 2

#print(_grid)

for i in range(0, 9):
    for j in range(0,9):
        org = (x, y)
        if _grid[j][i] > 0:
            image = cv2.putText(image, str(grid[j][i]), org, font, fontScale, color_black, thickness, cv2.LINE_AA)
        else:
            image = cv2.putText(image, str(grid[j][i]), org, font, fontScale, color_green, thickness, cv2.LINE_AA)
        y += 75
    x += 75
    y = 60

cv2.imwrite("output_sudoku" + ".jpg", image)

#plt.imshow(image)
#plt.show()
