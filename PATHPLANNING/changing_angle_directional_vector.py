"""
    DESCRIPTION: aide
"""

import math as m
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use("fivethirtyeight")


def smooth_directional_vector(points: list):
    points = np.array(points)
    vecteur = np.array([points[0], points[-1]], dtype=np.float)
    angle = m.atan((vecteur[1, 1]-vecteur[0, 1])/(vecteur[1, 0]-vecteur[0, 0]))
    print("INITIAL ANGLE RADIAN :", angle)
    print("INITIAL ANGLE DEGREE :", angle * 180 / m.pi)
    nombre_pts = points.shape[0]
    # l = (((points[-1, 0]-points[0, 0])**2) +
    #      ((points[-1, 1]-points[0, 1])**2))**0.5

    # y = 0
    print("CHANGING IN PROCESS....\n")

    # region BO GOSS VERSION
    # show_erreur = None
    # while y < 2000:
    #     y+=1

    #     # create points
    #     points_ligne = np.zeros((nombre_pts,2))
    #     d_x = (vecteur[1,0]-vecteur[0,0]) / (nombre_pts-1)
    #     d_y = (vecteur[1,1]-vecteur[0,1]) / (nombre_pts-1)
    #     for i in list(range(nombre_pts)):
    #         points_ligne[i,:] = np.array([[vecteur[0,0]+(d_x*i),vecteur[0,1]+(d_y*i)]])

    #     erreur_total = 0
    #     for i in list(range(nombre_pts)):
    #         erreur_total += (((points_ligne[i,0]-points[i,0])**2)+((points_ligne[i,1]-points[i,1])**2))**0.5
    #     erreur_moyenne = erreur_total/nombre_pts
    #     print("resultat", angle, " error ", erreur_moyenne)

    #     angle += 0.005

    #     # recompute points d'arriver with new angle.
    #     vecteur[1,0] = l*m.cos(angle)
    #     vecteur[1,1] = l*m.sin(angle)

    #     if show_erreur is None:
    #         show_angle = np.array([[angle-0.005]])
    #         show_erreur = np.array([[erreur_moyenne]])
    #     else:
    #         show_angle = np.concatenate((show_angle, np.array([[angle-0.005]])),axis=0)
    #         show_erreur = np.concatenate((show_erreur, np.array([[erreur_moyenne]])), axis=0)

    # x = np.linspace(0,2,show_erreur.shape[0])
    # fig = plt.plot(facecolor="white", figsize=(14.0,9.5))
    # #plt.set_aspect('equal', 'datalim')
    # plt.scatter(show_angle,show_erreur[:,0], s=4, c='black', label='error')
    # plt.show()
    # endregion

    # LOSER VERSION
    somme_x = 0
    somme_y = 0
    for i in list(range(nombre_pts)):
        somme_x += points[i, 0]
        somme_y += points[i, 1]
    angle_randian = m.atan(somme_y/somme_x)

    if angle_randian < 0:
        print("Radian angle is negative")
        print("ANGLE RADIAN :", angle_randian)
        angle_randian = angle_randian + np.pi
        angle_degre = angle_randian * 180 / m.pi
        print(angle_degre)
        # FINAL_ANGLE = angle_degre - (2*(angle_degre - 90))
        FINAL_ANGLE = angle_degre

    else:
        print("NEW ANGLE RADIAN :", angle_randian)
        angle_degre = angle_randian * 180 / m.pi
        FINAL_ANGLE = angle_degre
        print("NEW ANGLE DEGREE :", FINAL_ANGLE)

    print()
    adjacent_lenght = vecteur[1, 0] - vecteur[0, 0]
    print("ADJACENT LENGHT", adjacent_lenght)
    print()

    print("######################")
    print("ANGLE RADIAN :", angle_randian)
    print("ANGLE DEGREE :", angle_degre)

    real_rotation_degre = 90 - angle_degre
    real_rotation_radian = real_rotation_degre * np.pi / 180
    print(real_rotation_degre, real_rotation_radian)

    hypotenuse_distance = adjacent_lenght / m.cos(real_rotation_radian)
    print(hypotenuse_distance)
    # oposite_lenght = m.tan(real_rotation_radian) * adjacent_lenght
    # print(oposite_lenght)
    # ypotenuse_lenght = adjacent_lenght / m.cos(0.366519)
    # print("OPOSITE LENGHT", oposite_lenght)

    print("\n######################")
    print("WANTED COORDINATES")
    Z = [round(hypotenuse_distance * m.cos(real_rotation_radian)),
         round(hypotenuse_distance * m.sin(real_rotation_radian))]

    # print(Z)
    return Z


if __name__ == "__main__":

    # UP RIGHT
    # points = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [0, 5],
    #     [1, 5],
    #     [2, 5],
    #     [3, 5],
    #     [4, 5],
    #     [5, 5]]) # represente

    # UP RIGHT
    # points = np.array([
    #     [1, 2],
    #     [1, 3],
    #     [1, 4],
    #     [1, 5],
    #     [2, 5],
    #     [3, 5],
    #     [4, 5],
    #     [5, 5]]) # represente

    # UP LEFT
    # points = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [0, 5],
    #     [-1, 5],
    #     [-2, 5],
    #     [-3, 5],
    #     [-4, 5],
    #     [-5, 5]])

    points = np.array([
        [70, 71],
        [70, 72],
        [70, 73],
        [70, 74],
        [70, 75],
        [70, 76],
        [70, 77],
        [70, 78],
        [70, 79],
        [71, 79],
        [72, 79],
        [73, 79],
        [74, 79],
        [75, 79]])

    # DOWN LEFT
    # points = np.array([
    #     [0,   0],
    #     [0,  -1],
    #     [0,  -2],
    #     [0,  -3],
    #     [0,  -4],
    #     [0,  -5],
    #     [-1, -5],
    #     [-2, -5],
    #     [-3, -5],
    #     [-4, -5],
    #     [-5, -5]])

    # DOWN RIGHT
    # points = np.array([
    #     [0,  0],
    #     [0, -1],
    #     [0, -2],
    #     [0, -3],
    #     [0, -4],
    #     [0, -5],
    #     [1, -5],
    #     [2, -5],
    #     [3, -5],
    #     [4, -5],
    #     [5, -5]])

    # for coord in points:
    #     matrix[coord[0], coord[1]] = 9

    # print("Matrix :\n", matrix)

    # plt.scatter(points[:,0], points[:,1])
    # plt.show()

    print(smooth_directional_vector(points))
