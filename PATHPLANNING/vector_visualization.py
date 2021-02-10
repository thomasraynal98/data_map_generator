import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

# fig = plt.figure()
# ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
# vectors = [[1, 1], [1, 0], [1, -1], [0, -1]] #It's vectors not coordinates !
# vectors = [[1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0]] #It's vectors not coordinates !
# vectors = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
# starting_coord = [70, 70]
# end_coordinates = [145, 105]
# line_points_Y = [70, 70, 71, 71, 71, 71, 70, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 71, 72, 73, 74, 75, 75, 75, 74, 75, 74, 74, 73, 73, 72, 72, 72, 73, 72, 72, 73, 73, 74, 74, 74, 74, 75, 75, 76, 76, 76, 77, 77, 77, 78, 79, 79, 80, 80, 81, 81, 82, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 85, 88, 88, 89, 89, 90, 90, 91, 91, 91, 91, 91, 91, 91, 92, 93, 94, 95, 95, 97, 98, 99, 99, 99, 100, 101, 101, 102, 102, 103, 103, 104, 104, 105]
# line_points_X = [70, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 105, 105, 106, 106, 107, 107, 107, 106, 107, 111, 112, 112, 113, 113, 113, 113, 114, 115, 115, 116, 116, 116, 117, 117, 118, 118, 118, 119, 120, 120, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 127, 128, 128, 129, 129, 130, 130, 131, 132, 132, 133, 133, 133, 134, 135, 135, 136, 136, 136, 137, 138, 139, 139, 140, 140, 141, 141, 141, 142, 143, 143, 144, 145]


# coordinatooo = []

#     for index, val in enumerate(line_points_X):
#         for idx, v in enumerate(line_points_Y):
#             idx += index
#             coordinatooo.append([val, line_points_Y[idx]])
#             break

#     starting_coord = [coordinatooo[0][0], coordinatooo[0][0]]
#     end_coordinates = smooth_directional_vector(coordinatooo)


def plot_vector(vectors : list, starting_coord : list, end_coordinates : list, line_points_X : list, line_points_Y: list, mapeeeuh : list):
    
    # starting_coord = [coordinatooo[0][0], coordinatooo[0][0]]
    # end_coordinates = smooth_directional_vector(coordinatooo)
    
    """
    0: 5  -> green walkable field
    1: 7  -> red wall
    2: 16 -> pastel gray high cost to go through
    3: 15 -> gray higher cost to go through
    9: 6 -> light blue path
    5: 3  -> actual postion
    """

    dct = {1: 7., 0: 5., 2: 16., 3: 15., 9: 1., 5: 3.}
    n = [[dct[i] for i in j] for j in mapeeeuh]
    # print(n)

    plt.clf()
    plt.imshow(n, cmap='tab20', vmin=1, vmax=20)
    # plt.pause(0.005)
    # # plt.show()
    # return plt.plot()

    arrows = np.array(vectors)

    origin_X = [starting_coord[0]]
    for index, coord in enumerate(vectors):
        if index < len(vectors) - 1:
            origin_X.append(origin_X[index] + coord[0])
    
    origin_Y = [starting_coord[1]]
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

    print("res:", res)

    print("angle:", angle_degree)

    distance = np.sqrt((res[0] - 0)**2 + (res[1] - 0)**2)

    # print(angle_degree)
    # print(distance)
    # save_to_file.append([angle_degree, distance])

    # save_to_file = np.array(save_filooo, dtype=int)


    plt.clf()
    # plt.quiver(*origin, arrows[:,0], arrows[:,1], angles='xy', scale_units='xy', scale=1)

    plt.quiver(origin_X[0], origin_Y[0], res[0], res[1], color=['b'], angles='xy', scale_units='xy', scale=1)

    plt.scatter((origin_X[0] + end_coordinates[0]) // 2, (origin_Y[0] + end_coordinates[1]) // 2, s=30)

    plt.plot(line_points_X, line_points_Y, 'y')

    plt.imshow(n, cmap='tab20', vmin=1, vmax=20)

    # print(origin_X[0], end_coordinates[0], origin_Y[0], end_coordinates[1])
    plt.xlim(0, 260)
    plt.ylim(180, 0)
    plt.pause(0.005)
    # plt.show()
    return plt.plot()
    # return angle_degree, distance
    
def test_plot(vectors : list, starting_coord : list, end_coordinates : list, line_points_X : list, line_points_Y: list, mapeeeuh : list):
    """
    0: 5  -> green walkable field
    1: 7  -> red wall
    2: 16 -> pastel gray high cost to go through
    3: 15 -> gray higher cost to go through
    9: 6 -> light blue path
    5: 3  -> actual postion
    """

    dct = {1: 7., 0: 5., 2: 16., 3: 15., 9: 1., 5: 3.}
    n = [[dct[i] for i in j] for j in mapeeeuh]
    # print(n)

    # plt.clf()
    # plt.imshow(n, cmap='tab20', vmin=1, vmax=20)
    # plt.pause(0.005)
    # # plt.show()
    # return plt.plot()

    arrows = np.array(vectors)

    origin_X = [starting_coord[0]]
    for index, coord in enumerate(vectors):
        if index < len(vectors) - 1:
            origin_X.append(origin_X[index] + coord[0])
    
    origin_Y = [starting_coord[1]]
    for index, coord in enumerate(vectors):
        if index < len(vectors) - 1:
            origin_Y.append(origin_Y[index] + coord[1])

    # origin = np.array([origin_X, origin_Y]) # origin point

    # SUM of all vectors to get the directional vector from 50 cm away. (sum of 10 vectors)
    res = list()
    for j in range(0, len(arrows[0])):
        tmp = 0
        for i in range(0, len(arrows)):
            tmp = tmp + arrows[i][j]
        res.append(tmp)

    print("QUIVER RESULTAT:", res)

    plt.clf()
    # plt.quiver(*origin, arrows[:,0], arrows[:,1], angles='xy', scale_units='xy', scale=1)

    # plt.quiver(origin_X[0], origin_Y[0], res[0], res[1], color=['b'], angles='xy', scale_units='xy', scale=1)

    # plt.scatter((origin_X[0] + end_coordinates[0]) // 2, (origin_Y[0] + end_coordinates[1]) // 2, s=30)

    plt.plot(line_points_X, line_points_Y, 'r')

    plt.imshow(n, cmap='tab20', vmin=1, vmax=20)

    # print(origin_X[0], end_coordinates[0], origin_Y[0], end_coordinates[1])
    plt.xlim(0, 260)
    plt.ylim(180, 0)
    plt.pause(0.005)
    # plt.show()
    return plt.plot()


# fig = plt.figure(1)
# plot_vector(vectors, starting_coord, end_coordinates, line_points_X, line_points_Y)

# fig = plt.figure(2)
# plot_vector(vectors, starting_coord)

# plot_vector(vectors, starting_coord)
# vectors = [[1, 1], [1, 0], [1, -1], [0, -1]] #It's vectors not coordinates !
# starting_coord = [3,3]
# plot_vector(vectors, starting_coord)

# plt.show()
