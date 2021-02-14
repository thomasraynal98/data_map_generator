import numpy as np

import heapq

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from pre_compute_path_FASTER import * 

from vector_visualization import plot_vector, test_plot


ELEVATOR = (60, 300)
SOFTLAB = (580, 130)
BIOLAB = (310, 70)
SALOON = (460, 290)

 

##############################################################################

# plot grid

##############################################################################


def matrix_change_2(low_map_array, extend_coeff: int):
    """
        DESCRIPTION: transform map and add security.
        INPUT:
            * low_map_array = numpy[nxm] get map with 0 (empty), 1 (full)
        OUTPUT:
            * safe_map = numpy[nxm] return map with 0 (empty), 1 (full), 2 (safe)
    """
    extend_coeff_i = extend_coeff
    extend_coeff_j = extend_coeff
    safe_map = np.zeros((low_map_array.shape[0],low_map_array.shape[1]))

    for i in list(range(low_map_array.shape[0])):
        for j in list(range(low_map_array.shape[1])):
            # each cell, draw a box all around.
            
            if low_map_array[i,j] == 1:
                # detect bordure.
                ii = -(extend_coeff//2)
                while ii < (extend_coeff//2):
                    jj = -(extend_coeff//2)
                    while jj < (extend_coeff//2):
                        if (i+ii >= 0 and i+ii < low_map_array.shape[0]) and (j+jj >= 0 and j+jj < low_map_array.shape[1]):
                            if (ii == 0) and (jj == 0):
                                # a now wall.
                                safe_map[i+ii,j+jj] = 1
                            elif (low_map_array[i+ii,j+jj] == 0):
                                # a safe border when case is empty
                                safe_map[i+ii,j+jj] = 2 # 2
                        jj += 1
                    ii += 1

    return safe_map
 

grid = np.array([

    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],

    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# name_minimap = "PATHPLANNING/result/map/map_1_HD.txt"

# print("NUMPY ARRAY")
# low_map = input_from_txt(name_minimap)

# print("low map:\n", low_map)

# grid = array_map_to_ZeroOne(low_map)

# print("map_zero_one:\n", grid)

# plot_map(grid)

# grid = matrix_change(grid, 12)
# plot_map(grid)
# print()

grid = matrix_change_2(grid, 12)


# start point and goal

# start = ELEVATOR

# goal = BIOLAB



start = (0,0)

goal = (0,19)


##############################################################################

# heuristic function for path scoring

##############################################################################

 

def heuristic(a, b):

    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

##############################################################################

# path finding function

##############################################################################


def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:

        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]

            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]: 
                    if array[neighbor[0]][neighbor[1]] == 2:
                        tentative_g_score += 10

            elif 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue

            else:
                continue
 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
 

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor)) 

route = astar(grid, start, goal)

route = route + [start]

route = route[::-1]

print(route)


##############################################################################

# plot the path

##############################################################################

 

#extract x and y coordinates from route list

x_coords = []

y_coords = []

for i in (range(0,len(route))):

    x = route[i][0]

    y = route[i][1]

    x_coords.append(x)

    y_coords.append(y)

# plot map and path

fig, ax = plt.subplots(figsize=(20,20))

ax.imshow(grid, cmap=plt.cm.Dark2)

ax.scatter(start[1], start[0], marker = "*", color = "yellow", s = 200)

ax.scatter(goal[1], goal[0], marker = "*", color = "red", s = 200)

ax.plot(y_coords, x_coords, color = "black")
plt.plot()
plt.pause(10)
print()


line_points_Y = [route[0][0]]
line_points_X = [route[0][1]]

vector_range = 20 # lenght of cells where the robot while create its directional vector
current_position = tuple(start)
curr_pos_fatlist = route.index(start)
# print(curr_pos_fatlist)
tempoo = route[curr_pos_fatlist : curr_pos_fatlist + vector_range]

vectors = []
for k in range(len(tempoo) - 1):
    # in security list we have (y,x) coordinate so I just changed it to (x,y) for creating vectors list
    vectors.append([tempoo[k+1][1] - tempoo[k][1], (tempoo[k+1][0] - tempoo[k][0])])

# print("VECTORS :", vectors)

fig = plt.figure(1)

# Middle point of tempoo which is list of points of path to draw the line followed by the robot.

line_points_X.append((tempoo[0][1] + tempoo[-1][1]) // 2)
line_points_Y.append((tempoo[0][0] + tempoo[-1][0]) // 2)

# plt.clf()
# plt.imshow(n, cmap='tab20', vmin=1, vmax=20)

arrows = np.array(vectors)

origin_X = [tempoo[0][1]]
for index, coord in enumerate(vectors):
    if index < len(vectors) - 1:
        origin_X.append(origin_X[index] + coord[0])

origin_Y = [tempoo[0][0]]
for index, coord in enumerate(vectors):
    if index < len(vectors) - 1:
        origin_Y.append(origin_Y[index] + coord[1])

# origin = np.array([origin_X, origin_Y]) # origin point

# SUM of all vectors to get the directional vector from 50 cm away. (sum of 10 vectors)
res = list()
for mj in range(0, len(arrows[0])):
    tmp = 0
    for mi in range(0, len(arrows)):
        tmp = tmp + arrows[mi][mj]
    res.append(tmp)

# print("QUIVER RESULTAT:", res)

if res[1] == 0:
    angle = 0
elif res[0] == 0:
    angle = 270
else:
    angle = np.arctan(res[1] / res[0])

angle_degree = 360 - (angle * 180 / np.pi)
angle_degree = angle_degree % 360

distance = np.sqrt((res[0] - 0)**2 + (res[1] - 0)**2)

# plt.clf()
# plt.quiver(*origin, arrows[:,0], arrows[:,1], angles='xy', scale_units='xy', scale=1)

# plt.quiver(origin_X[0], origin_Y[0], res[0], res[1], color=['b'], angles='xy', scale_units='xy', scale=1)

# plt.scatter((origin_X[0] + tempoo[-1][1]) // 2, (origin_Y[0] + tempoo[-1][0]) // 2, s=30)

# plt.plot(line_points_X, line_points_Y, 'y')

# plt.imshow(grid, cmap='tab20', vmin=1, vmax=20)

# # print(origin_X[0], end_coordinates[0], origin_Y[0], end_coordinates[1])
# plt.xlim(0, 260)
# plt.ylim(180, 0)
# plt.pause(5)

# return plt.plot()


# plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, grid)

plt.show()