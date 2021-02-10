from pre_compute_path_FASTER import * 

import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import heapq

EMPTY = 0
OBSTACLE = 255

def import_image_and_transform(path):

    #variable
    seuil = 150

    image = cv2.imread(path,0)
    la_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    for i in list(range(image.shape[0])):
        for j in list(range(image.shape[1])):
            if image[i,j] < seuil:
                la_map[i,j] = OBSTACLE
            else:
                la_map[i,j] = EMPTY

    return la_map

# map_name = "PATHPLANNING/result/image/big_map/bigmap_1.png"
# matrix = import_image_and_transform(map_name)
# name = "PATHPLANNING/result/map/bigmap.txt"
# with open(name, 'w') as f:
#     for item in matrix:
#         for i in range(len(item)):
#             if i == len(matrix[0]) - 1:
#                 f.write("%s " % item[i])
#                 f.write("\n")
#             else:
#                 f.write("%s " % item[i])

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

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

                    if array[neighbor[0]][neighbor[1]] == 1:

                        continue

                else:

                    # array bound y walls

                    continue

            else:

                # array bound x walls

                continue
 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):

                continue
 

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:

                came_from[neighbor] = current

                gscore[neighbor] = tentative_g_score

                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                heapq.heappush(oheap, (fscore[neighbor], neighbor))



name_minimap = "PATHPLANNING/result/map/bigmap.txt"

print("NUMPY ARRAY")
low_map = input_from_txt(name_minimap)

print("low map:\n", low_map)

map_zero_one = array_map_to_ZeroOne(low_map)

print("map_zero_one:\n", map_zero_one)

plot_map(map_zero_one)
print()

start_Y = input("Please write the X coordinate of your starting point : ")
start_X = input("Please write the Y coordinate of your starting point : ")
end_Y = input("Please write the X coordinate of your endind point : ")
end_X = input("Please write the Y coordinate of your endind point : ")
start = [int(start_X), int(start_Y)]
end = [int(end_X), int(end_Y)]

start = []


route = astar(low_map, start, goal)

print(route)






if low_map[end[0]][end[1]] == 1:
    print("Problem end value (equal 1)")

if low_map[start[0]][start[1]] != 1:
    print("\nSTART :", start)
    print("END :", end, "\n")
    map_path, path = Astar_from_map(map_zero_one, start, end)


print()