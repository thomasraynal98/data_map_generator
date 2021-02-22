import heapq
import math
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import figure
from scipy.stats import norm

EXTEND_AREA = 10.0  # [m] grid map extention length

show_animation = True

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

plt.figure(figsize=(5,5))
plt.imshow(grid)
plt.axis("equal")
plt.title('First map')
plt.show()

index_wall_X, index_wall_Y = np.where(grid==1)

print(f"X Index:{index_wall_X},\n\nY Index:{index_wall_Y}")
print()
print(f"GRID SHAPE:{grid.shape}") 


def generate_gaussian_grid_map(ox, oy, xyreso, std):
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    gmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        for iy in range(yw):

            x = ix * xyreso + minx
            y = iy * xyreso + miny

            # Search minimum distance
            mindis = float("inf")
            for (iox, ioy) in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if mindis >= d:
                    mindis = d

            pdf = (1.0 - norm.cdf(mindis, 0.0, std))
            gmap[ix][iy] = pdf
    return gmap, minx, maxx, miny, maxy

def calc_grid_map_config(ox, oy, xyreso):
    # minx = round(min(ox)) # - EXTEND_AREA / 2.0)
    # miny = round(min(oy)) # - EXTEND_AREA / 2.0)
    # maxx = round(max(ox)) # + EXTEND_AREA / 2.0)
    # maxy = round(max(oy)) # + EXTEND_AREA / 2.0)
    minx = 0
    miny = 0
    maxx = 20
    maxy = 20
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw

##############################################################################

# plot grid

##############################################################################

xyreso = 1  # xy grid resolution
STD = 5.0  # standard diviation for gaussian distribution

#for i in range(5):
    # ox = np.around((np.random.rand(4) - 0.5) * 10.0)
    # oy = np.around((np.random.rand(4) - 0.5) * 10.0)
ox = index_wall_X
oy = index_wall_Y

gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(
    ox, oy, xyreso, STD)

grid = np.array(gmap)
# print(grid)

plt.figure(figsize=(5,5))
plt.imshow(grid, cmap=plt.cm.Blues)
plt.axis("equal")
plt.title('Gaussian map')
plt.show()

start = (0,0)

goal = (0,19)

############################

# RAPPEL : ox and oy are walls coordinates

############################

grid[ox, oy] = 1

plt.figure(figsize=(5,5))
plt.imshow(grid, cmap=plt.cm.Blues)
plt.axis("equal")
plt.plot(0, 0, "xr")
plt.plot(19, 0, "ob")
plt.title('Augmented Wall score to 1')
plt.show()

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
            tentative_g_score = (gscore[current] + heuristic(current, neighbor)) + (grid[current])

            # print(f"GSCORE OF CURRENT POSITION {gscore[current]}")
            
            # if 0 <= neighbor[0] < array.shape[0]:
            #     if 0 <= neighbor[1] < array.shape[1]: 
            #         if array[neighbor[0]][neighbor[1]] == 2:
            #             tentative_g_score += 10

            if 0 <= neighbor[0] < array.shape[0]:
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

print(f"PATH:{route}")


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

print()
print(f"X coordinates:{x_coords}, \n\nY coordinates:{y_coords}")

# plot map and path

fig, ax = plt.subplots(figsize=(5,5))

ax.imshow(grid, cmap='Blues')

ax.scatter(start[1], start[0], marker = "*", color = "yellow", s = 200)

ax.scatter(goal[1], goal[0], marker = "*", color = "red", s = 200)

ax.plot(y_coords, x_coords, color = "black")
plt.title('Pathfinding through gaussian map')
plt.plot()
plt.show()