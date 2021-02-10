from changing_angle_directional_vector import smooth_directional_vector
from pre_compute_path_FASTER import * 
from vector_visualization import plot_vector, test_plot

import ast
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def import_image_and_transform(path):
    #variable
    seuil = 150

    image = cv2.imread(path,0)
    la_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    for i in list(range(image.shape[0])):
        for j in list(range(image.shape[1])):
            if image[i,j] < seuil:
                la_map[i,j] = 1
            else:
                la_map[i,j] = 0

    return la_map

obstacle_mode = False
mode_camera = False
multi_map = True

# CHANGE THE OBSTACLE TO A SQUARE 20 TIMES BIGGER 
class Obstacle:
    def __init__(self, position : list):
        self.position = position
    
    def size_it(self, real_size : int, security_rangeuu : int, robot_size : int):
        """
        real_size : 
                for 1 meter cell grid, size would be 20 to get a 5cm cell grid. With 20 you have 10 cm on each size of the obstacle
        security_rangeuu : 
                distance between robot and wall
        robot_size : 
                60cm x 60cm for exemple
        """

        size = real_size // 2 + 1
        robot = robot_size // 2 + 1
        # list_obstacle = []
        # for i in range(self.position[0] - real_size, self.position[0] + real_size + 1):
        #     for j in range(self.position[1] - real_size, self.position[1] + real_size + 1):
        #         list_obstacle.append((i, j))

        list_obstacle = [ 
            (i,j) for i in range(self.position[0] - size - security_rangeuu - robot, self.position[0] + size + security_rangeuu + robot) 
                  for j in range(self.position[1] - size - security_rangeuu - robot, self.position[1] + size + security_rangeuu + robot)
        ]

        return list_obstacle

if multi_map == True:
    # "FAKE" Dijkstra's return:
    dijkstra_list = ['1', '2', '3']
    # instead of (4, 4) give CAMERA's COORDINATE
    HD_pathfinding(dijkstra_list, 4, 4)

    line_list = []
    name = "PATHPLANNING/result/path/list_path_result_HD.txt"
    f = open(name,"r")
    lines = f.readlines()
    line_list.append(lines)
    f.close()

    paths = [ ast.literal_eval(line_list[r][c]) for r, row in enumerate(line_list) for c, ele in enumerate(row) ] 
    #str to list, tuple, dict whatever it looks like
    list_path = [[list(elem) for elem in path] for path in paths]

    # list_path : list of list with coords of low path 
    # for path in list_path:
    #     print(path)

if multi_map == False:
    # Pathfinding for only one map
    window_name = "mini_map_1"
    map_name = "PATHPLANNING/result/image/big_map/bigmap_1.png"
    # map_name = "PATHPLANNING/result/image/LOW/minimap_LOW_1.jpg"

    image = cv2.imread(map_name)
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

    start_X = input("Please write the X coordinate of your starting point : ")
    start_Y = input("Please write the Y coordinate of your starting point : ")
    end_X = input("Please write the X coordinate of your endind point : ")
    end_Y = input("Please write the Y coordinate of your endind point : ")
    start = [int(start_X), int(start_Y)]
    end = [int(end_X), int(end_Y)]

    # Stock paths from all maps 
    list_path_map = []

    # Stock extend paths from all maps 
    sum_extend = []

    # marre = import_image_and_transform(map_name)

    # matrix = np.array(map_path)

    # name = "PATHPLANNING/result/map/result/bigmap_result.txt"
    
    # with open(name, 'w') as f:
    #     for item in matrix:
    #         for i in range(len(item)):
    #             if i == len(map_path[0]) - 1:
    #                 f.write("%s " % item[i])
    #                 f.write("\n")
    #             else:
    #                 f.write("%s " % item[i])

    # name_minimap = "PATHPLANNING/big_map/" + map_number + ".txt"
    name_minimap = "PATHPLANNING/result/map/map_1_L.txt"

    print("NUMPY ARRAY")
    low_map = input_from_txt(name_minimap)
    
    print("low map:\n", low_map)
    
    map_zero_one = array_map_to_ZeroOne(low_map)

    print("map_zero_one:\n", map_zero_one)
    
    if map_zero_one[end[0]][end[1]] == 1:
        print("Problem end value (equal 1)")
        # break

    if map_zero_one[start[0]][start[1]] != 1:
        print("\nSTART :", start)
        print("END :", end, "\n")
        map_path, path = Astar_from_map(map_zero_one, start, end)
        
        # Map with path written :
        for row in map_path:
            print(row)

        print("\nPath is : ", path, "\n")
        list_path_map.append(path)
    
    else:
        print("Start isn't walkable")


    move = [[-1, 0 ], # go up
            [ 0, -1], # go left
            [ 1, 0 ], # go down
            [ 0, 1 ]]

    extend_path_temp = []

    for coord in path:
        for new_position in move:
            a = coord[0] + new_position[0]
            b = coord[1] + new_position[1]
            c = [a, b]
            if a >= 0 and b >= 0:
                extend_path_temp.append(c)

    b_set = set(tuple(x) for x in extend_path_temp)
    extend_path = [ list(x) for x in b_set ]

    path = [list(ele) for ele in path] 

    new_path = [p for p in extend_path if p not in path]

    new_path = [coord for coord in new_path if coord[0] <= len(map_path)-1 and coord[1] <= len(map_path[0])-1]

    print("EXTEND without dup and path :", new_path)
    sum_extend.append(new_path)

    print("MAP LENGHT :", len(map_path[0]))

    for coord in new_path:
        if map_path[coord[0]][coord[1]] != 1:
            map_path[coord[0]][coord[1]] = 2
    
    height = 0
    for i in map_path:
        height += 1

    for i in range(height):
        for j in range(len(map_path[0])):
            if map_path[i][j] == 0:
                map_path[i][j] = 3

    for row in map_path:
        print(row)

    plot_map(map_path)

start_time = time.time()

if mode_camera == False:
    if multi_map == True:
        # We go throught the several maps (for the exemple there is 3 maps)
        x = 0
        while x < len(list_path):

            if x == len(list_path): # There is len(list_path)-1 maps so if this condition is true BREAK EVERYTHING
                print("YOU HAVE FINISH YOUR JOURNEY")
                input("please press any key to finish")
                break

            map_number = x + 1
            name_minimap = "PATHPLANNING/result/map/result/map_" + str(map_number) + "_result_HD.txt"
            # low_map = LOW_map_from_txt(name_minimap)
            low_map = input_from_txt(name_minimap)
            print("FIRST time printing Low Map")
            print(low_map)
            plot_map(low_map)

            # The obstacle is common to the same map
            # print("Creating STATIC obstacle......")
            if obstacle_mode == True:
                coord_x = input("Give me obstacle height coord : ")
                coord_y = input("Give me obstacle lenght coord : ")
                obstacle = Obstacle((int(coord_x),int(coord_y)))
                obstacle_list = obstacle.size_it(20, 0, 0)
            # for obsta in obstacle_list:
            #     print(low_map[obsta[0], obsta[1]])

            # Iterate through the waypoints list
            nb_list_waypoints = 0
            line_points_Y = [list_path[x][0][0][0]]
            line_points_X = [list_path[x][0][0][1]]

            while nb_list_waypoints <= len(list_path[x]):
                print("Go through first map:", time.time() - start_time)

                print("################ Waypoint list number", str(nb_list_waypoints), "#################")
                if nb_list_waypoints == len(list_path[x]): # There is len(list_path[x]) waypoint lists. If we went through all of them chaange map !
                    x = x + 1
                    break

                goal = list_path[x][nb_list_waypoints][-1]
                current_position = list_path[x][nb_list_waypoints][0]

                
                # for val in low_map:
                #     print(val)

                # low_map = np.array(low_map)
                # print(low_map)

                # plot_map(low_map)

                start_time = time.time()

                to_save_file = []
                while current_position != goal:
                    for i in range(len(list_path[x][nb_list_waypoints])):
                        # First position in path : initialize
                        if obstacle_mode == True:
                            for obsta in obstacle_list:
                                low_map[obsta[0], obsta[1]] = 1
                        
                        current_position = list_path[x][nb_list_waypoints][i] # path[i] == [3, 3] ou [3, 4] ou etc...
                        # print(current_position)
                        # Creating list of path coord in SECURITY zone
                        security_range = 20
                        security_list = []

                        
                        if i == 0:
                            low_map[current_position[0], current_position[1]] = 5

                            if obstacle_mode == True:
                                for obsta in obstacle_list:
                                    low_map[obsta[0], obsta[1]] = 1
                                
                            # plot_map(low_map)

                            if low_map[goal[0], goal[1]] == 1:
                                i = 0
                                ind = 1
                                # Iterate over list_path[x][nb_list_waypoints + 1] to find new goal after the current goal in the path list
                                while low_map[goal[0], goal[1]] != 9:
                                    goal = list_path[x][nb_list_waypoints + ind][i]
                                    i = i + 1
                                    if i >= len(list_path[x][nb_list_waypoints + ind]):
                                        i = 0
                                        ind = ind + 1
                                # If ind > 1 it means that a all list of waypoints was jupped, so we can delete it
                                if ind > 1:
                                    position_initial = nb_list_waypoints + 1
                                    position_end = nb_list_waypoints + ind
                                    for val in range(position_initial, position_end):
                                        list_path[x].remove(list_path[x][val])


                                # Goal has changed so we need to compute a new path between current and new goal
                                temp = np.where(low_map == 5) 
                                low_map[temp] = 0

                                low_map, path = Astar_from_map(low_map, current_position, goal)

                                low_map[path[0][0], path[0][1]] = 5

                                list_path[x][nb_list_waypoints] = path
                                
                                
                                for val in list_path[x][nb_list_waypoints + 1]:
                                    low_map[val[0], val[1]] = 0

                                # plot_map(low_map)
                                # IF OBJECTIF IS COVERING goal cell, we need to chose goal from next nb_list_waypoint and then adjust
                                next_waypoint_path = [list(elem) for elem in list_path[x][nb_list_waypoints + 1]]
                                # goal = list(goal)
                                # print(next_waypoint_path)

                                # next_goal = next_waypoint_path[-1]
                                # print(next_goal)

                                # Create a new waypoint path because goal was changed !
                                next_waypoint_path = [ next_waypoint_path[index:] for index, e in enumerate(next_waypoint_path) if e == list(goal) ]
                                next_waypoint_path = [item for sublist in next_waypoint_path for item in sublist] # flatten list of list to list
                                # print(next_waypoint_path)
                                next_waypoint_path = [tuple(elem) for elem in next_waypoint_path]
                                goal = tuple(goal)
                                # print(goal)
                                list_path[x][nb_list_waypoints + 1] = next_waypoint_path
                                
                                for val in next_waypoint_path:
                                    low_map[val[0], val[1]] = 9

                                break

                                # plot_map(low_map)################


                                
                            # plot_map(low_map)
                            # print("ARRAY MAP")
                            # print(low_map)
                            # print()
                        # Next position in path : moving through the pass
                        # Osbtacle mooving in the same time
                        # First obstacle moves and then current position udpates or recalculates wheither the next postion is an obstacle or not
                        else:
                            print("CURRENT CURRRENT", current_position)
                            if current_position == goal:
                                low_map[current_position[0], current_position[1]] = 0
                                nb_list_waypoints = nb_list_waypoints + 1
                                
                                
                                # old_position = list_path[x][nb_list_waypoints][i-1]
                                # low_map[old_position[0], old_position[1]] = 0
                                # low_map[current_position[0], current_position[1]] = 5
                                
                                # print("current_position == goal")
                                # plot_map(low_map)
                                break

                            else:
                                # CHECK IF OBSTACLE IN TRAJECTORY. If yes then replanning path and break
                                vector_range = 20 # lenght of cells where the robot while create its directional vector
                                # print("#############")
                                # print("CURRENT POS:", current_position)
                                # print(current_position[0], current_position[1])
                                # print(list_path[x])
                                mouak = list_path[x]
                                testouille = [item for mouak in mouak for item in mouak] # flatten list of list to list
                                # print(testouille)

                                # tempoo = [ [i, j] for i in range(current_position[0] - vector_range, current_position[0] + vector_range) for j in range(current_position[1] - vector_range, current_position[1] + vector_range) if i < len(low_map) and j < len(low_map[0]) and i > 0 and j > 0 and low_map[i, j] == 9]
                                curr_pos_fatlist = testouille.index(current_position)
                                # print(curr_pos_fatlist)
                                tempoo = testouille[curr_pos_fatlist : curr_pos_fatlist + vector_range]
                                # print(tempoo)

                                ######################################
                                # Trying to make the path trace avoid obstacles
                                limit_i_up = []
                                limit_j_up = []
                                for titi in range(current_position[0] - vector_range, current_position[0]):
                                    for tjtj in range(current_position[1] - vector_range, current_position[1] + vector_range):
                                        if titi < len(low_map) and tjtj < len(low_map[0]) and titi > 0 and tjtj > 0 and low_map[titi, tjtj] == 1 and current_position[1] - tjtj == 0:# and low_map[i,j] == 9:
                                            limit_i_up.append(titi)
                                            limit_j_up.append(tjtj)

                                limit_i_down = []
                                limit_j_down = []
                                for titi in range(current_position[0], current_position[0] + vector_range):
                                    for tjtj in range(current_position[1] - vector_range, current_position[1] + vector_range):
                                        if titi < len(low_map) and tjtj < len(low_map[0]) and titi > 0 and tjtj > 0 and low_map[titi, tjtj] == 1 and current_position[1] - tjtj == 0:# and low_map[i,j] == 9:
                                            limit_i_down.append(titi)
                                            limit_j_down.append(tjtj)

                                limit_i_left = []
                                limit_j_left = []
                                for titi in range(current_position[0] - vector_range, current_position[0] + vector_range):
                                    for tjtj in range(current_position[1] - vector_range, current_position[1]):
                                        if titi < len(low_map) and tjtj < len(low_map[0]) and titi > 0 and tjtj > 0 and low_map[titi, tjtj] == 1 and current_position[0] - titi == 0:# and low_map[i,j] == 9:
                                            limit_i_left.append(titi)
                                            limit_j_left.append(tjtj)

                                limit_i_right = []
                                limit_j_right = []
                                for titi in range(current_position[0], current_position[0] + vector_range):
                                    for tjtj in range(current_position[1], current_position[1] + vector_range):
                                        if titi < len(low_map) and tjtj < len(low_map[0]) and titi > 0 and tjtj > 0 and low_map[titi, tjtj] == 1 and current_position[0] - titi == 0:# and low_map[i,j] == 9:
                                            limit_i_right.append(titi)
                                            limit_j_right.append(tjtj)

                                # # Working on this specific case : obstacle under current position
                                if limit_i_down and current_position[0] not in limit_i_down and current_position[0] < limit_i_down[0]:
                                    testonca = []
                                    for valuux in tempoo:
                                        # print(valuux)
                                        for yaaaa in range(valuux[0], valuux[0] + vector_range):
                                            # print(yaaaa)
                                            # print(low_map[yaaaa, valuux[1]])
                                            if low_map[yaaaa, valuux[1]] == 1:
                                                # print("ok")
                                                testonca.append((valuux[0], valuux[1]))
                                                break

                                    # print(testonca)
                                    # print()
                                    last_pos_fatlist = testouille.index(testonca[-1])

                                    if limit_j_down and current_position[1] == limit_j_down[-1]:
                                        # tempoo = [ [i, j] for i in range(current_position[0] - vector_range, limit_i_down[0]) for j in range(current_position[1] - vector_range, current_position[1] + vector_range) if i < len(low_map) and j < len(low_map[0]) and i > 0 and j > 0 and low_map[i,j] == 9]
                                        # X is an mooving offset that represente the lenght between current point and the obstacle right angle
                                        tempoo = testouille[curr_pos_fatlist : last_pos_fatlist + 2]
                                        limit_j_down = []

                                # # Working on this specific case : obstacle over current position
                                elif limit_i_up and current_position[0] not in limit_i_up and current_position[0] > limit_i_up[-1]:
                                    testonca = []
                                    for valuux in tempoo:
                                        # print(valuux)
                                        for yaaaa in range(valuux[0] - vector_range, valuux[0]):
                                            # print(yaaaa)
                                            # print(low_map[yaaaa, valuux[1]])
                                            if low_map[yaaaa, valuux[1]] == 1:
                                                # print("ok")
                                                testonca.append((valuux[0], valuux[1]))
                                                break

                                    last_pos_fatlist = testouille.index(testonca[-1])

                                    if limit_j_up and current_position[1] == limit_j_up[-1]:
                                        tempoo = testouille[curr_pos_fatlist : last_pos_fatlist + 2]
                                        limit_j_up = []

                                # # Working on this specific case : obstacle on the left of current position
                                elif limit_j_left and current_position[1] not in limit_j_left and current_position[1] > limit_j_left[-1]:
                                    
                                    testonca = []
                                    for valuux in tempoo:
                                        # print(valuux)
                                        for yaaaa in range(valuux[1] - vector_range, valuux[1]):
                                            # print(yaaaa)
                                            # print(low_map[yaaaa, valuux[1]])
                                            if low_map[valuux[0], yaaaa] == 1:
                                                # print("ok")
                                                testonca.append((valuux[0], valuux[1]))
                                                break

                                    last_pos_fatlist = testouille.index(testonca[-1])

                                    if limit_i_left and current_position[0] == limit_i_left[-1]:
                                        tempoo = testouille[curr_pos_fatlist : last_pos_fatlist + 2]
                                        limit_i_left = []
                                    
                                # Working on this specific case : obstacle on the right of current position
                                elif limit_j_right and current_position[1] not in limit_j_right and current_position[1] < limit_j_right[0]:
                                    
                                    testonca = []
                                    for valuux in tempoo:
                                        # print(valuux)
                                        for yaaaa in range(valuux[1], valuux[1] + vector_range):
                                            # print(yaaaa)
                                            # print(low_map[yaaaa, valuux[1]])
                                            if low_map[valuux[0], yaaaa] == 1:
                                                # print("ok")
                                                testonca.append((valuux[0], valuux[1]))
                                                break

                                    last_pos_fatlist = testouille.index(testonca[-1])
                                    
                                    if limit_i_right and current_position[0] == limit_i_right[-1]:
                                        tempoo = testouille[curr_pos_fatlist : last_pos_fatlist + 2]
                                        limit_i_right = []
                                
                                ######################################

                                else:
                                    # tempoo = [ [i, j] for i in range(current_position[0] - vector_range, current_position[0] + vector_range) for j in range(current_position[1] - vector_range, current_position[1] + vector_range) if i < len(low_map) and j < len(low_map[0]) and i > 0 and j > 0 and low_map[i,j] == 9]
                                    tempoo = testouille[curr_pos_fatlist : curr_pos_fatlist + vector_range]
                                # print("PATH LIST:\n", tempoo)
                                # tempoo = np.array(tempoo)

                                # smooth_end = smooth_directional_vector(tempoo)

                                # print("SMOOTH END", smooth_end)
                                # print()
                                
                                # print(len(vectors))
                                # List of coord belonging to path and in a range of 20 cells
                                # security_list = [ list_path[x][nb_list_waypoints][j] for j in range(i+1, len(list_path[x][nb_list_waypoints])) if list_path[x][nb_list_waypoints][j][0] <= list_path[x][nb_list_waypoints][i][0] + security_range and list_path[x][nb_list_waypoints][j][1] <= list_path[x][nb_list_waypoints][i][1] + security_range]
                                # print("SECURITY LIST:\n", security_list)
                                # print(len(security_list))
                                vectors = []
                                for k in range(len(tempoo) - 1):
                                    # print(security_list[i])
                                    # in security list we have (y,x) coordinate so I just changed it to (x,y) for creating vectors list
                                    vectors.append([tempoo[k+1][1] - tempoo[k][1], (tempoo[k+1][0] - tempoo[k][0])])
                                
                                # print("VECTORS :", vectors)

                                fig = plt.figure(1)
                                # print("COORD [0] [1] :", tempoo[0][0], tempoo[0][1])
                                # print(tempoo[0][-1])

                                # Middle point of tempoo which is list of points of path to draw the line followed by the robot.

                                line_points_X.append((tempoo[0][1] + tempoo[-1][1]) // 2)
                                line_points_Y.append((tempoo[0][0] + tempoo[-1][0]) // 2)
                                # print(tempoo[0])
                                # test_plot(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)

                                # print("Vectors :", vectors, "\nLine Point X", line_points_X, "\nLine Point Y", line_points_Y)
                                # angle_degree, distance = plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)
                                plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)

                                # plot_vector(vectors, line_points_X, line_points_Y, low_map)
                                # plt.scatter((tempoo[0][1] + tempoo[-1][1]) // 2, (tempoo[0][0] + tempoo[-1][0]) // 2, s=30)
                                # plt.plot(line_points_X, line_points_Y)
                                
                                # to_save_file.append([angle_degree, distance])
                                # yaraay = np.array(to_save_file, dtype=int)
                                # print(yaraay)
                                # np.save('result_angle_dist', yaraay)
                                plt.show()
                                # # print("curren pos:", current_position)
                                # plot_map(low_map)


                                #region CHECK IF OBSTACLE IN TRAJECTORY. If yes then replanning path and break

                                # List of coord belonging to path and in a range of 20 cells
                                security_list = [ list_path[x][nb_list_waypoints][j] for j in range(i+1, len(list_path[x][nb_list_waypoints])) if list_path[x][nb_list_waypoints][j][0] <= list_path[x][nb_list_waypoints][i][0] + security_range and list_path[x][nb_list_waypoints][j][1] <= list_path[x][nb_list_waypoints][i][1] + security_range]

                                # print(security_list)

                                # print("curren pos:", current_position)
                                # plot_map(low_map)
                                if obstacle_mode == True:
                                    if any(coord in obstacle_list for coord in security_list):
                                        
                                        # Here we want to change values only between current and goal coordinates

                                        temp = [ (i,j) for i in range(len(low_map)) for j in range(len(low_map[i])) if [i, j] != goal and low_map[i,j] == 9]

                                        # print(temp)
                                        for value in range(len(temp)):
                                            low_map[temp[value]] = 0

                                        temp = np.where(low_map == 5) 
                                        low_map[temp] = 0

                                        # for val in low_map:
                                        #     print(val)
                                        # print("ARRAY MAP")
                                        # print(low_map)
                                        # print("Before A*")
                                        # plot_map(low_map)
                                        
                                        low_map, path = Astar_from_map(low_map, current_position, goal)
                                        
                                        for obsta in obstacle_list:
                                            low_map[obsta[0], obsta[1]] = 1
                                        
                                        plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)

                                        plt.show()
                                        
                                        # print("before changin")
                                        # print(low_map)
                                        # low_map = np.array(low_map)
                                        # print("after changin")
                                        # print(low_map)

                                        # print(path)
                                        
                                        # Set current position to 5
                                            
                                        low_map[path[0][0], path[0][1]] = 5
                                        
                                        # print("After A* low map:")
                                        # plot_map(low_map)

                                        # print("NEW low map")
                                        # for val in low_map:
                                        #     print(val)
                                        # print("ARRAY MAP")
                                        # print(low_map)

                                        # plot_map(low_map)
                                        list_path[x][nb_list_waypoints] = path
                                        break
                                    
                                    #endregion

                                #region No Obstacle on the road chief ! Maybe the robot has drifted ! CHECK IT
                                # DRIFT simulation change next position meaning that position could be a cell around the real current_position
                                # last_position = list(list_path[x][nb_list_waypoints][i-1])

                                # move = [[ 1, 0 ], # go down
                                        # [ 0, 1 ], # go right
                                        # [ -1, 0 ], # go up
                                        # [ 0, -1 ], # go left
                                        # [-1, -1],
                                        # [ 1, -1],
                                        # [ -1, 1],
                                        # [ 1, 1 ]]

                                # move_down =    [[ 1, 0 ],
                                #                 [ 1, -1],
                                #                 [ 1, 1 ]]

                                # move_right =   [[ 0, 1 ],
                                #                 [ 1, 1 ],
                                #                 [ -1, 1]]
                                
                                # move_left =    [[ 0, -1 ],
                                #                 [ 1, -1 ],
                                #                 [ -1, -1]]
                                
                                # move_up  =     [[ -1, 0 ],
                                #                 [ -1, -1],
                                #                 [ -1, 1 ]]
                                
                                # if [list(current_position)[0] - last_position[0], list(current_position)[1] - last_position[1]] == [1, 0]:
                                #     move = move_down

                                # if [list(current_position)[0] - last_position[0], list(current_position)[1] - last_position[1]] == [-1, 0]:
                                #     move = move_up

                                # if [list(current_position)[0] - last_position[0], list(current_position)[1] - last_position[1]] == [0, 1]:
                                #     move = move_right

                                # if [list(current_position)[0] - last_position[0], list(current_position)[1] - last_position[1]] == [0, -1]:
                                #     move = move_left

                                # WARNING : if low_map[mov] == 1 for mov in movin_to: remove it from list
                                # movin_to = [ [last_position[0] + mov[0], last_position[1] + mov[1]] for mov in move ]
                                # for val in movin_to:
                                #     #####################################################
                                #     # index 180 is out of bounds for axis 0 with size 180
                                #     #####################################################
                                #     if low_map[val[0], val[1]] == 1:
                                #         movin_to.remove(val)

                                # val = 0
                                # for index, ele in enumerate(movin_to):
                                #     if ele == list(current_position):
                                #         val = index
                                #         break
                                # weights = [ 0.6 if i == val else 0.2 for i in range(len(movin_to)) ]

                                # new_current_position = random.choices(
                                #     population=movin_to,
                                #     weights=weights,
                                #     k=1
                                # )
                                # new_current_position = tuple([item for new_current_position in new_current_position for item in new_current_position]) # flatten list of list to list
                                
                                
                                # print("wanted position:", current_position, "actual postion:", new_current_position)
                                # plot_map(low_map)

                                # NO DRIFTING if current position is one step before goal because arf problems ! Goal is reach but nb_list_waypoint isn't updated
                                #region DRIFTING ACTIVED
                                # if current_position != list_path[x][nb_list_waypoints][-2]:
                                #     if new_current_position != current_position:
                                #         # print("THERE IS SOME DRIFTING CHIEF")
                                #         # low_map[new_current_position[0], new_current_position[1]] = 5
                                #         # print(np.where(low_map == 9))
                                #         # plot_map(low_map)
                                #         # print("goal", goal)

                                #         # temp = [ path[:index + 1] for index, e in enumerate(path) if e == goal ]
                                #         # list_path = [item for sublist in list_path for item in sublist] # flatten list of list to list

                                #         # temp = [ (i,j) for i in range(len(low_map)) for j in range(len(low_map[i])) if i <= goal[0] and j <= goal[1] and low_map[i ,j] == 9]

                                #         # print(temp)
                                #         for value in range(len(list_path[x][nb_list_waypoints])):
                                #             low_map[list_path[x][nb_list_waypoints][value]] = 0
                                        
                                #         temp = np.where(low_map == 5) 
                                #         low_map[temp] = 0
                                        
                                #         # print("Now everything is deleted till next goal !")
                                #         # plot_map(low_map)
                                #         # start_time = time.time()
                                #         low_map, path = Astar_from_map(low_map, new_current_position, goal)
                                #         # print("########## TIME TIME TIME #########", time.time() - start_time)
                                #         # Set current position to 5
                                #         # temp = np.where(low_map == 5) 
                                #         # low_map[temp] = 0
                                            
                                #         low_map[path[0][0], path[0][1]] = 5
                                #         # print("remapped")
                                #         # fig = plt.figure(1)
                                #         # plot_map(low_map)##########
                                #         # plt.show()
                                        
                                #         list_path[x][nb_list_waypoints] = path
                                #         break
                                    #endregion
                                #endregion
                                
                                # print(path)
                                old_position = list_path[x][nb_list_waypoints][i-1]

                                low_map[old_position[0], old_position[1]] = 0
                                low_map[current_position[0], current_position[1]] = 5
                                
                                # Plot to see each step !
                                # fig = plt.figure(1)
                                # plot_map(low_map)########################
                                # plt.show()

                                # print("ARRAY MAP")
                                
                                # print()
                                # print("################ NEW plot ############")
                                # plot_map(low_map)

if mode_camera == True:
    if multi_map == True:
        x = 0
        while x < len(list_path):

            if x == len(list_path): # There is len(list_path)-1 maps so if this condition is true BREAK EVERYTHING
                print("YOU HAVE FINISH YOUR JOURNEY")
                input("please press any key to finish")
                break

            map_number = x + 1
            name_minimap = "PATHPLANNING/result/map/result/map_" + str(map_number) + "_result_HD.txt"
            # low_map = LOW_map_from_txt(name_minimap)
            low_map = input_from_txt(name_minimap)
            print("FIRST time printing Low Map")
            print(low_map)
            plot_map(low_map)

            # Iterate through the waypoints list
            nb_list_waypoints = 0
            line_points_Y = [list_path[x][0][0][0]]
            line_points_X = [list_path[x][0][0][1]]

            while nb_list_waypoints <= len(list_path[x]):
                print("Go through first map:", time.time() - start_time)

                print("################ Waypoint list number", str(nb_list_waypoints), "#################")
                if nb_list_waypoints == len(list_path[x]): # There is len(list_path[x]) waypoint lists. If we went through all of them change map !
                    x = x + 1
                    break

                goal = list_path[x][nb_list_waypoints][-1]
                current_position = list_path[x][nb_list_waypoints][0]
                ################################
                # current_position = CAMERA POSITION
                ################################

                while current_position != goal:
                    ##############################################
                    # ROBOT SUPPOSED TO MOVE HERE
                    # FIRST WE PUSH THE BOX OURSELVES 
                    ##############################################

                    
                    # current_position = ROBOT_POS

                    # Current position may have change or smth in the way : compute a new path
                    temp = np.where(low_map == 5) 
                    low_map[temp] = 0

                    low_map, path = Astar_from_map(low_map, current_position, goal)

                    low_map[path[0][0], path[0][1]] = 5

                    list_path[x][nb_list_waypoints] = path

                    # print("PATH RECALCULATED")

                    if low_map[goal[0], goal[1]] == 1:
                        i = 0
                        ind = 1
                        
                        # Iterate over list_path[x][nb_list_waypoints + 1] to find new goal after the current goal in the path list
                        while low_map[goal[0], goal[1]] != 9:
                            goal = list_path[x][nb_list_waypoints + ind][i]
                            i = i + 1
                            if i >= len(list_path[x][nb_list_waypoints + ind]):
                                i = 0
                                ind = ind + 1
                        
                        # If ind > 1 it means that a all list of waypoints was jupped, so we can delete it
                        if ind > 1:
                            position_initial = nb_list_waypoints + 1
                            position_end = nb_list_waypoints + ind
                            for val in range(position_initial, position_end):
                                list_path[x].remove(list_path[x][val])


                        # Goal has changed so we need to compute a new path between current and new goal
                        temp = np.where(low_map == 5) 
                        low_map[temp] = 0

                        low_map, path = Astar_from_map(low_map, current_position, goal)

                        low_map[path[0][0], path[0][1]] = 5

                        list_path[x][nb_list_waypoints] = path

                        for val in list_path[x][nb_list_waypoints + 1]:
                            low_map[val[0], val[1]] = 0

                        # IF OBJECTIF IS COVERING goal cell, we need to chose goal from next nb_list_waypoint and then adjust
                        next_waypoint_path = [list(elem) for elem in list_path[x][nb_list_waypoints + 1]]

                        # Create a new waypoint path because goal was changed !
                        next_waypoint_path = [ next_waypoint_path[index:] for index, e in enumerate(next_waypoint_path) if e == list(goal) ]
                        next_waypoint_path = [item for sublist in next_waypoint_path for item in sublist] # flatten list of list to list

                        next_waypoint_path = [tuple(elem) for elem in next_waypoint_path]
                        goal = tuple(goal)

                        list_path[x][nb_list_waypoints + 1] = next_waypoint_path
                        
                        for val in next_waypoint_path:
                            low_map[val[0], val[1]] = 9

                        break

                    if current_position == goal:
                        low_map[current_position[0], current_position[1]] = 0
                        nb_list_waypoints = nb_list_waypoints + 1
                        
                        
                        # old_position = list_path[x][nb_list_waypoints][i-1]
                        # low_map[old_position[0], old_position[1]] = 0
                        # low_map[current_position[0], current_position[1]] = 5
                        
                        # print("current_position == goal")
                        # plot_map(low_map)
                        break
                    
                    else :
                        vector_range = 20 # lenght of cells where the robot while create its directional vector
                        mouak = list_path[x]
                        testouille = [item for mouak in mouak for item in mouak] # flatten list of list to list
                        # print(testouille)

                        # tempoo = [ [i, j] for i in range(current_position[0] - vector_range, current_position[0] + vector_range) for j in range(current_position[1] - vector_range, current_position[1] + vector_range) if i < len(low_map) and j < len(low_map[0]) and i > 0 and j > 0 and low_map[i, j] == 9]
                        curr_pos_fatlist = testouille.index(current_position)
                        # print(curr_pos_fatlist)
                        tempoo = testouille[curr_pos_fatlist : curr_pos_fatlist + vector_range]

                        vectors = []
                        for k in range(len(tempoo) - 1):
                            # in security list we have (y,x) coordinate so I just changed it to (x,y) for creating vectors list
                            vectors.append([tempoo[k+1][1] - tempoo[k][1], (tempoo[k+1][0] - tempoo[k][0])])
                        
                        # print("VECTORS :", vectors)

                        fig = plt.figure(1)

                        # Middle point of tempoo which is list of points of path to draw the line followed by the robot.

                        line_points_X.append((tempoo[0][1] + tempoo[-1][1]) // 2)
                        line_points_Y.append((tempoo[0][0] + tempoo[-1][0]) // 2)
                        
                        plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)

                        plt.show()

    if multi_map == False:
        current_position = start
        ################################
        # current_position = CAMERA POSITION
        ################################
        line_points_Y = [list_path_map[0][0][0]]
        line_points_X = [list_path_map[0][0][1]]

        while current_position != end:
            ##############################################
            # ROBOT SUPPOSED TO MOVE HERE
            # FIRST WE PUSH THE BOX OURSELVES 
            ##############################################


            # current_position = ROBOT_POS

            # Current position may have change or smth in the way : compute a new path
            temp = np.where(low_map == 5) 
            low_map[temp] = 0

            low_map, path = Astar_from_map(low_map, current_position, end)

            low_map[path[0][0], path[0][1]] = 5

            list_path_map = path

            # print("PATH RECALCULATED")

            # if low_map[end[0], end[1]] == 1:
                
            #     # Iterate over list_path[x][nb_list_waypoints + 1] to find new goal after the current goal in the path list
            #     while low_map[end[0], end[1]] != 9:
            #         end = list_path[x][nb_list_waypoints + ind][i]
            #         i = i + 1
            #         if i >= len(list_path[x][nb_list_waypoints + ind]):
            #             i = 0
            #             ind = ind + 1
                
            #     # If ind > 1 it means that a all list of waypoints was jupped, so we can delete it
            #     if ind > 1:
            #         position_initial = nb_list_waypoints + 1
            #         position_end = nb_list_waypoints + ind
            #         for val in range(position_initial, position_end):
            #             list_path[x].remove(list_path[x][val])


            #     # Goal has changed so we need to compute a new path between current and new goal
            #     temp = np.where(low_map == 5) 
            #     low_map[temp] = 0

            #     low_map, path = Astar_from_map(low_map, current_position, end)

            #     low_map[path[0][0], path[0][1]] = 5

            #     list_path[x][nb_list_waypoints] = path

            #     for val in list_path[x][nb_list_waypoints + 1]:
            #         low_map[val[0], val[1]] = 0

            #     # IF OBJECTIF IS COVERING goal cell, we need to chose goal from next nb_list_waypoint and then adjust
            #     next_waypoint_path = [list(elem) for elem in list_path[x][nb_list_waypoints + 1]]

            #     # Create a new waypoint path because goal was changed !
            #     next_waypoint_path = [ next_waypoint_path[index:] for index, e in enumerate(next_waypoint_path) if e == list(end) ]
            #     next_waypoint_path = [item for sublist in next_waypoint_path for item in sublist] # flatten list of list to list

            #     next_waypoint_path = [tuple(elem) for elem in next_waypoint_path]
            #     end = tuple(end)

            #     list_path[x][nb_list_waypoints + 1] = next_waypoint_path
                
            #     for val in next_waypoint_path:
            #         low_map[val[0], val[1]] = 9

            #     break

            if current_position == end:
                print("YOU ACHIEVED YOUR GOAL")
                # low_map[current_position[0], current_position[1]] = 0
                # nb_list_waypoints = nb_list_waypoints + 1
                
                
                # old_position = list_path[x][nb_list_waypoints][i-1]
                # low_map[old_position[0], old_position[1]] = 0
                # low_map[current_position[0], current_position[1]] = 5
                
                # print("current_position == goal")
                # plot_map(low_map)
                break

            else :
                vector_range = 3 # lenght of cells where the robot while create its directional vector
                current_position = tuple(current_position)
                curr_pos_fatlist = list_path_map.index(current_position)
                # print(curr_pos_fatlist)
                tempoo = list_path_map[curr_pos_fatlist : curr_pos_fatlist + vector_range]

                vectors = []
                for k in range(len(tempoo) - 1):
                    # in security list we have (y,x) coordinate so I just changed it to (x,y) for creating vectors list
                    vectors.append([tempoo[k+1][1] - tempoo[k][1], (tempoo[k+1][0] - tempoo[k][0])])
                
                # print("VECTORS :", vectors)

                fig = plt.figure(1)
                
                # Middle point of tempoo which is list of points of path to draw the line followed by the robot.

                line_points_X.append((tempoo[0][1] + tempoo[-1][1]) // 2)
                line_points_Y.append((tempoo[0][0] + tempoo[-1][0]) // 2)
                
                plot_vector(vectors, [tempoo[0][1], tempoo[0][0]], [tempoo[-1][1], tempoo[-1][0]], line_points_X, line_points_Y, low_map)

                plt.show()
