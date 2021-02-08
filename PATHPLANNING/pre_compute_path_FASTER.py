import cv2
from dijkstra_graph import main_dijkstra
from map_modifier_HD import matrix_change
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ion()

#region A* CODE

class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position

# Return the path of the search function
def return_path(current_node, maze):
    path = []
    # no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    # result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    # result = [[maze[i] for i in len(maze) for j in len(maze[i])]]

    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    # start_value = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        maze[path[i][0]][path[i][1]] = 9
        # start_value += 1

    return maze, path

def search(maze, start, end):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param start:
        :param end:
        :return:
    """

    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = []

    # Add the start node
    yet_to_visit_list.append(start_node)

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . search movement is left-right-top-bottom
    #(4 movements) from every positon

    move  =  [[-1, 0 ], # go up
              [ 0, -1], # go left
              [ 1, 0 ], # go down
              [ 0, 1 ]]# go right
            #   [-1, -1],
            #   [ 1, -1],
            #   [ -1, 1],
            #   [ 1, 1 ]]


    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perform Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent

            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    #find maze has got how many rows and columns
    no_rows, no_columns = np.shape(maze)

    # Loop until you find the ends a result, the algorithm can consider long "jumps" along straight (horizontal, vertical and diagonal) lines in the grid, rather than the small steps from one grid position to the next that ordinary A* considers.[2]

    while len(yet_to_visit_list) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1
        cost = 1


        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outer_iterations > max_iterations:
            print ("giving up on pathfinding too many iterations")
            return return_path(current_node, maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []
        for new_position in move:

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or
                node_position[0] < 0 or
                node_position[1] > (no_columns -1) or
                node_position[1] < 0):
                continue

            # Make sure walkable terrain for neighboors. If un walkable, node not created
            if maze[node_position[0]][node_position[1]] == 1:
                continue
            # if maze[node_position[0]][node_position[1]] == 2:
            #     cost = 30
            # if maze[node_position[0]][node_position[1]] == 3:
            #     cost = 80

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost
            # print(cost)
            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)

#endregion

def create_random_map(width, empty_pourcentage):
    """
        Returns a map randomly filled with 0 and 1 
        :param width:
        :param empty_pourcentage:
    """

    # If random map we have to make sure start and end initiate on empty cells
    # new_map[0][0] = 0
    # new_map[width - 1][width - 1] = 0

    empty = (width * empty_pourcentage) // 100
    obstacle = width - empty

    new_map = []

    for i in range(width):
        new_map.append([0]*empty + [1]*obstacle)
    for e in new_map:
        random.shuffle(e)

    return new_map

def LOW_map_from_txt(path : str) -> list:
    """
        Returns a list of list from the path file to create a map
        :param path:
    """

    line_list = []

    f = open(path,"r")
    lines = f.readlines()
    line_list.append(lines)
    f.close()

    low_map = []
    for i in range(len(line_list)):
        for j in range(len(line_list[i])):
            test = line_list[i][j].split(' ')
            test = test[:-1]
            low_map.append(test)

    for i in range(len(low_map)):
        for j in range(len(low_map[i])):
            low_map[i][j] = int(low_map[i][j])

    # print ("Modified list is : " + str(low_map))

    # for x in low_map:
    #     print(x)

    return low_map

def input_from_txt(path : str) -> list:
    """
        Returns a list of list from the path file to create a map
        :param path:
    """
    low_map = np.loadtxt(path, dtype=int)
    # print(input)    

    return low_map

def map_to_ZeroOne(mapeeeuh : list) -> list:
    """
        Returns a map filled with 0 and 1 instead of having 1 and 255
        :param map:
    """
    # ZERO : empty
    # ONE : obstacle

    for i in range(len(mapeeeuh)):
        for j in range(len(mapeeeuh[i])):
            # print(mapeeeuh[i][j])
            if mapeeeuh[i][j] == 0:
                mapeeeuh[i][j] = 1
            if mapeeeuh[i][j] == 255:
                mapeeeuh[i][j] = 0

    return mapeeeuh

def array_map_to_ZeroOne(mapeeeuh):
    """
        Returns a map filled with 0 and 1 instead of having 1 and 255
        :param map:
    """
    # ZERO : empty
    # ONE : obstacle

    # for i in range(len(mapeeeuh)):
    #     for j in range(len(mapeeeuh[i])):
    #         # print(mapeeeuh[i][j])
    #         if mapeeeuh[i][j] == 0:
    #             mapeeeuh[i][j] = 1
    #         if mapeeeuh[i][j] == 255:
    #             mapeeeuh[i][j] = 0
    
    temp_1 = np.where(mapeeeuh == 0)
    temp_2 = np.where(mapeeeuh == 255)
    mapeeeuh[temp_1] = 1
    mapeeeuh[temp_2] = 0
    
    return mapeeeuh

def plot_map(mapeeeuh : list):
    # dct permet d'étaler les couleurs de sismic de 1 à 20
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
    plt.pause(0.0005)
    # plt.show()
    return plt.plot()

def Astar_from_map(mapeuuuh : list, start : list, end : list):
    
    path = search(mapeuuuh, start, end)

    # for x in path:
    #     print(x)

    return path

def low_pathfinding(dijkstra_list : list):
    
    line_list = []

    # Stock paths from all maps 
    list_path_map = []

    # Stock extend paths from all maps 
    sum_extend = []

    for index, i in enumerate(dijkstra_list):
        # First case : First low map so we need to initiate the starting point 
        
        if index == 0:
            map_number = i
            name = "result/link/link_" + map_number + "_L.txt"
            f = open(name,"r")
            lines = f.readlines()
            line_list.append(lines)
            f.close()

            connection_map = []
            for i in range(len(line_list)):
                for j in range(len(line_list[i])):
                    test = line_list[i][j].split(' ')
                    if len(test) == 6:
                        test = test[:-1]
                    elif len(test) == 5:
                        for e in range(len(test)):
                            if e == 4:
                                test[e] = test[e][0]
                            
                    connection_map.append(test)

            window_name = "mini_map_" + str(map_number)
            map_name = "result/image/LOW/minimap_LOW_" + map_number + ".jpg"
            image = cv2.imread(map_name)
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, image)

            # print("Link from map", map_number, ":", connection_map, "\n")
            line_list = []
            

            lenght_connection_map = len(connection_map) - 1
            # print(lenght_connection_map)
            random_nb = random.randint(0, lenght_connection_map)

            # print(random_nb)
            # print(connection_map[random_nb])

            end = connection_map[random_nb][:2]
            # print(end)

            reversed_end = end[::-1]
            for i in range(len(reversed_end)):
                reversed_end[i] = int(reversed_end[i])

            for i in range(len(end)):
                end[i] = int(end[i])   
        
            cv2.waitKey(0)

            start_X = input("Please write the X coordinate of your starting point : ")
            start_Y = input("Please write the Y coordinate of your starting point : ")

            start = [int(start_X), int(start_Y)]

            # print("Start coordinates :", start)
            # print("End coordinates :", reversed_end)


            name_minimap = "result/map/map_" + map_number + "_L.txt"
            # low_map = LOW_map_from_txt(name_minimap)
            # print("low map:\n", low_map)

            # map_zero_one = map_to_ZeroOne(low_map)

            # print("map_zero_one:\n", map_zero_one)

            # print()
            print("NUMPY ARRAY")
            low_map = input_from_txt(name_minimap)
            
            print("low map:\n", low_map)
            
            map_zero_one = array_map_to_ZeroOne(low_map)

            print("map_zero_one:\n", map_zero_one)

            # for row in map_zero_one:
            #     print(row)

            # print("\nValue of map at start coord :", map_zero_one[start[0]][start[1]])
            # print("Value of map at end coord :", map_zero_one[end[0]][end[1]])
            
            if map_zero_one[end[0]][end[1]] == 1:
                print("Problem end value (equal 1)")
                break

            # for row in map_zero_one:
            #     print(row)
            # print("\nNumber of rows : ", len(map_zero_one))
            # print("Number of columns : ", len(map_zero_one[0]), "\n")


            # print(map_zero_one)

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
                    # [-1, -1],
                    # [ 1, -1],
                    # [ -1, 1],
                    # [ 1, 1 ]]

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

            # print("EXTEND :", extend_path)

            # print("PATH :", path)
            path = [list(ele) for ele in path] 

            # for e in extend_path:
            #     print("E",e)
            #     for b in path:
            #         print("B",b)
            new_path = [p for p in extend_path if p not in path]
             # print(new_path)

            new_path = [coord for coord in new_path if coord[0] <= len(map_path)-1 and coord[1] <= len(map_path[0])-1]

            # for nijo in no_way:
            #   print("NIJO",nijo)

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
                        # print(map_path[i][j])
                        map_path[i][j] = 3
                        # print(map_path[i][j])

            print()

            for row in map_path:
                print(row)

            plot_map(map_path)

            matrix = np.array(map_path)

            name = "result/map/result/map_" + str(map_number) + "_result_L.txt"
            
            with open(name, 'w') as f:
                for item in matrix:
                    for i in range(len(item)):
                        if i == len(map_path[0]) - 1:
                            f.write("%s " % item[i])
                            f.write("\n")
                        else:
                            f.write("%s " % item[i])

            old_end = end

            # print('old end', old_end)

            connection_map = []

        elif index == len(dijkstra_list)-1:
            map_number = i
            name = "result/link/link_" + map_number + "_L.txt"
            f = open(name,"r")
            lines = f.readlines()
            line_list.append(lines)
            f.close()

            connection_map = []
            for i in range(len(line_list)):
                for j in range(len(line_list[i])):
                    test = line_list[i][j].split(' ')
                    if len(test) == 6:
                        test = test[:-1]
                    elif len(test) == 5:
                        for e in range(len(test)):
                            if e == 4:
                                test[e] = test[e][0]
                            
                    connection_map.append(test)

            for i in range(len(connection_map)):
                for j in range(len(connection_map[i])):
                    connection_map[i][j] = int(connection_map[i][j])

            window_name = "mini_map_" + str(map_number)
            map_name = "result/image/LOW/minimap_LOW_" + map_number + ".jpg"
            image = cv2.imread(map_name)
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, image)

            # print("Link from map", map_number, ":", connection_map, "\n")
            line_list = []
                    
            cv2.waitKey(0)

            end_Y = input("Please write the X coordinate of your goal : ")
            end_X = input("Please write the Y coordinate of your goal : ")

            end = [int(end_X), int(end_Y)]


            ######### GET GOOD COORDINATES ! ##############
            # Start is the coordinate linked with the end from step i-1

            start = old_end

            for i in connection_map:
                if start[0] == i[2] and start[1] == i[3]:
                    # print("replace start by i[0] and i[1]", i[0], i[1], i[2], i[3]) 
                    start[0] = i[1]
                    start[1] = i[0]

            ###############################################
            
            reversed_start = start[::-1]
            for i in range(len(reversed_start)):
                reversed_start[i] = int(reversed_start[i])
           
            # print("Start coordinates :", reversed_start)
            # print("Goal coordinates :", end)

            name_minimap = "result/map/map_" + map_number + "_L.txt"
            low_map = LOW_map_from_txt(name_minimap)

            map_zero_one = map_to_ZeroOne(low_map)

            # for row in map_zero_one:
            #     print(row)

            # print("\nValue of map at start coord :", map_zero_one[reversed_start[0]][reversed_start[1]])
            # print("Value of map at end coord :", map_zero_one[end[0]][end[1]])

            cost = 1

            if map_zero_one[reversed_start[0]][reversed_start[1]] != 1:
                print("\nSTART :", reversed_start)
                print("END :", end, "\n")
                map_path, path = Astar_from_map(map_zero_one, reversed_start, end)

                # Map with path written :
                for row in map_path:
                    print(row)

                print("\nPath is : ", path, "\n")
                list_path_map.append(path)
            else:
                print("Start isn't walkable")

            old_end = end
            connection_map = []

            move = [[-1, 0 ], # go up
                    [ 0, -1], # go left
                    [ 1, 0 ], # go down
                    [ 0, 1 ],
                    [-1, -1],
                    [ 1, -1],
                    [ -1, 1],
                    [ 1, 1 ]]

            extend_path_temp = []

            for coord in path:
                for new_position in move:
                    a = coord[0] + new_position[0]
                    b = coord[1] + new_position[1]
                    c = [a,b]
                    if a >= 0 and b >= 0:
                        extend_path_temp.append(c)

            b_set = set(tuple(x) for x in extend_path_temp)
            extend_path = [ list(x) for x in b_set ]

            # print("EXTEND :", extend_path)

            # print("PATH :", path)
            path = [list(ele) for ele in path] 

            # for e in extend_path:
            #     print("E",e)
            #     for b in path:
            #         print("B",b)
            new_path = [p for p in extend_path if p not in path]
             # print(new_path)

            new_path = [coord for coord in new_path if coord[0] <= len(map_path)-1 and coord[1] <= len(map_path[0])-1]


            # for nijo in no_way:
            #   print("NIJO",nijo)

            print("EXTEND without dup and path :", new_path)
            sum_extend.append(new_path)

            for coord in new_path:
                if map_path[coord[0]][coord[1]] != 1:
                    map_path[coord[0]][coord[1]] = 2

            height = 0
            for i in map_path:
                height +=1

            for i in range(height):
                for j in range(len(map_path[0])):
                    if map_path[i][j] == 0:
                        # print(map_path[i][j])
                        map_path[i][j] = 3
                        # print(map_path[i][j])

            print()

            for row in map_path:
                print(row)

            plot_map(map_path)

            matrix = np.array(map_path)

            name = "result/map/result/map_" + str(map_number) + "_result_L.txt"
            
            with open(name, 'w') as f:
                for item in matrix:
                    for i in range(len(item)):
                        if i == len(map_path[0]) - 1:
                            f.write("%s " % item[i])
                            f.write("\n")
                        else:
                            f.write("%s " % item[i])

            print()

            # for papa in list_path_map:
            #     print(papa)

            # for papa in sum_extend:
            #     print(papa)


            matrix = np.array(list_path_map)

            name = "result/path/list_path_result_L.txt"
            
            with open(name, 'w') as f:
                for item in matrix:
                        f.write("%s\n" % item)

            return list_path_map, sum_extend

            
            # for index, path_map in enumerate(list_path_map):
            #     print("For low map", index, "the path is :", path_map)
            # print("Sum of all paths", list_path_map)
            # print()

            # move = [[-1, 0 ], # go up
            #         [ 0, -1], # go left
            #         [ 1, 0 ], # go down
            #         [ 0, 1 ],
            #         [-1, -1],
            #         [ 1, -1],
            #         [ -1, 1],
            #         [ 1, 1 ]]

            # temp = []
            # extend_path = []

            # for index, path in enumerate(list_path_map):
            #     # print("Map number", index, "of final path :", path)
            #     for coord in path:
            #         # print("coord", coord)
            #         for new_position in move:
            #             a = coord[0] + new_position[0]
            #             b = coord[1] + new_position[1]
            #             c = [a,b]
            #             if a >= 0 and b >= 0:
            #                 temp.append(c)
            #     extend_path.append(temp)
            #     temp = []

            # print()

            # extend_without_duplicate = []
            # for path in extend_path:
            #     b_set = set(tuple(x) for x in path)
            #     path = [ list(x) for x in b_set ]
            #     extend_without_duplicate.append(path)
            
            # # print("extend_path")

            # # for index, e in enumerate(extend_path):
            # #     print("Map number", index, "'s extend path :", e, "\n")

            # # print("extend_without_duplicate")

            # # for index, e in enumerate(extend_without_duplicate):
            # #     print("Map number", index, "'s extend path :", e, "\n")
            
            # # print("\nextend_without_duplicateW\n")
            # # for e in extend_without_duplicate:
            # #     print(e)

            # # print("\nFINALE_PATH\n")
            # # for e in list_path_map:
            # #     print(e)

            # # print()
            # # print()

            # for i in range(len(list_path_map)):
            #     list_path_map[i] = [list(ele) for ele in list_path_map[i]]

            # path_surrounding = []

            # for index, e in enumerate(extend_without_duplicate):
            #     for index_2, b in enumerate(list_path_map):
            #         if index == index_2:
            #             print("list_path_map", index, ":", b)
            #             new_path = [path for path in e if path not in b]
            #             path_surrounding.append(new_path)

            # print()

            # for index, nijo in enumerate(path_surrounding):
            #     print("map", index, "path extended :", nijo)                    

        else:
            map_number = i
            name = "result/link/link_" + map_number + "_L.txt"
            f = open(name,"r")
            lines = f.readlines()
            line_list.append(lines)
            f.close()

            connection_map = []
            for i in range(len(line_list)):
                for j in range(len(line_list[i])):
                    test = line_list[i][j].split(' ')
                    if len(test) == 6:
                        test = test[:-1]
                    elif len(test) == 5:
                        for e in range(len(test)):
                            if e == 4:
                                test[e] = test[e][0]
                            
                    connection_map.append(test)

            for i in range(len(connection_map)):
                for j in range(len(connection_map[i])):
                    connection_map[i][j] = int(connection_map[i][j])

            # print("Link from map", map_number, ":", connection_map, "\n")
            line_list = []
            
            ############ FIND START ############
            start = old_end

            for i in connection_map:
                if start[0] == i[2] and start[1] == i[3]:
                    # print("replace start by i[0] and i[1]", i[0], i[1], i[2], i[3]) 
                    start[0] = i[1]
                    start[1] = i[0]
            
            reversed_start = start[::-1]
            # for i in range(len(reversed_start)):
            #     reversed_start[i] = int(reversed_start[i])

            # print("reserved_start :",reversed_start)
            # print("start :", start)

            ############ FIND GOAL ############
            connection_next_map = []
            for i in connection_map:
                if i[len(i)-1] == 3:
                    connection_next_map.append(i)

            # print(connection_next_map)
            lenght_next_connection_map = len(connection_next_map) - 1
            random_nb = random.randint(0, lenght_next_connection_map)
            # print(random_nb)

            # print(random_nb)
            # print(connection_next_map[random_nb])
            end = connection_next_map[random_nb][:2]
            # print("end :", end)

            reversed_end = end[::-1]

            # print("Start coordinates :", start)
            # print("Goal coordinates :", reversed_end)
            name_minimap = "result/map/map_" + map_number + "_L.txt"
            low_map = LOW_map_from_txt(name_minimap)

            map_zero_one = map_to_ZeroOne(low_map)

            # for row in map_zero_one:
            #     print(row)

            # print("\nValue of map at start coord :", map_zero_one[reversed_start[0]][reversed_start[1]])
            # print("Value of map at end coord :", map_zero_one[end[0]][end[1]])

            cost = 1

            if map_zero_one[reversed_start[0]][reversed_start[1]] != 1:
                print("\nSTART :", reversed_start)
                print("GOAL :", end, "\n")
                map_path, path = Astar_from_map(map_zero_one, reversed_start, end)

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
                    [ 0, 1 ],
                    [-1, -1],
                    [ 1, -1],
                    [ -1, 1],
                    [ 1, 1 ]]

            extend_path_temp = []

            for coord in path:
                for new_position in move:
                    a = coord[0] + new_position[0]
                    b = coord[1] + new_position[1]
                    c = [a,b]
                    if a >= 0 and b >= 0:
                        extend_path_temp.append(c)

            b_set = set(tuple(x) for x in extend_path_temp)
            extend_path = [ list(x) for x in b_set ]

            # print("EXTEND :", extend_path)

            # print("PATH :", path)
            path = [list(ele) for ele in path] 

            # for e in extend_path:
            #     print("E",e)
            #     for b in path:
            #         print("B",b)
            new_path = [p for p in extend_path if p not in path]
             # print(new_path)
            new_path = [coord for coord in new_path if coord[0] <= len(map_path)-1 and coord[1] <= len(map_path[0])-1]

            # for nijo in no_way:
            #   print("NIJO",nijo)

            print("EXTEND without dup and path :", new_path)
            sum_extend.append(new_path)

            for coord in new_path:
                if map_path[coord[0]][coord[1]] != 1:
                    map_path[coord[0]][coord[1]] = 2

            height = 0
            for i in map_path:
                height +=1

            for i in range(height):
                for j in range(len(map_path[0])):
                    if map_path[i][j] == 0:
                        # print(map_path[i][j])
                        map_path[i][j] = 3
                        # print(map_path[i][j])

            print()

            for row in map_path:
                print(row)

            plot_map(map_path)

            matrix = np.array(map_path)

            name = "result/map/result/map_" + str(map_number) + "_result_L.txt"
            
            with open(name, 'w') as f:
                for item in matrix:
                    for i in range(len(item)):
                        if i == len(map_path[0]) - 1:
                            f.write("%s " % item[i])
                            f.write("\n")
                        else:
                            f.write("%s " % item[i])

            old_end = end
            connection_map = []

def HD_pathfinding(dijkstra_list : list):
    
    print("\nHD_pathfinding")

    path_low, extend_path_low = low_pathfinding(dijkstra_list)

    for i in range(len(path_low)):
        path_low[i] = [list(ele) for ele in path_low[i]]
    
    print("RECAP of low pathfinding returns")
    print("PATH LOW")
    for row in path_low:
        print(row)
    
    print("EXTEND PATH LOW")
    for row in extend_path_low:
        print(row)

    to_save_txt = []

    for index_dij, i in enumerate(dijkstra_list):
        map_number = i

        #region HD MAP IMPORT

        name_hdmap = "result/map/map_" + str(map_number) + "_HD.txt"

        hd_map = LOW_map_from_txt(name_hdmap)

        map_zero_one_hd = map_to_ZeroOne(hd_map)
        plot_map(map_zero_one_hd)
        



        ##################################################################
        # COEFF TO EXTEND OBSTACLES SIZE ! 
        # If robot is 60x60 cm and we want a wall security range of 30cm we need to set the coeff to 12 
        # because we have 12 cells of 5cm
        map_zero_one_hd = matrix_change(map_zero_one_hd, 12)
        plot_map(map_zero_one_hd)
        ##################################################################





        #endregion

        for index, path in enumerate(extend_path_low):
            if index == index_dij:
                for rows in path:
                    for ii in range(rows[0] * 20, rows[0] * 20 + 20):
                        for jj in range(rows[1] * 20, rows[1] * 20 + 20):
                            if map_zero_one_hd[ii][jj] != 1:
                                # print("not equal to zero")
                                map_zero_one_hd[ii][jj] = 2

        path_low_modulo2 = []

        for index, path in enumerate(path_low):
            if index == index_dij:
                if index == len(dijkstra_list)-1:
                    print("index = dijkstra")
                    path_temp = [i for index, i in enumerate(path) if index%2 == 0]
                    if len(path) % 2 != 0:
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            elif rows == path_temp[0]:
                                ii = rows[0] * 20
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                        #     if map_zero_one_hd[ii][jj] != 1:
                        #         # print("not equal to zero")
                        #         map_zero_one_hd[ii][jj] = 'X'
                    else:
                        path_temp.append(path[-1])
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            elif rows == path_temp[0]:
                                ii = rows[0] * 20
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                
                elif index == 0:
                    path_temp = [i for index, i in enumerate(path) if index%2 == 0]
                    if len(path) % 2 != 0:
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 19
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                        #     if map_zero_one_hd[ii][jj] != 1:
                        #         # print("not equal to zero")
                        #         map_zero_one_hd[ii][jj] = 'X'
                    else:
                        path_temp.append(path[-1])
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 19
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                
                else:
                    path_temp = [i for index, i in enumerate(path) if index%2 == 0]
                    if len(path) % 2 != 0:
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 19
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            
                            elif rows == path_temp[0]:
                                ii = rows[0] * 20
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                            
                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                        #     if map_zero_one_hd[ii][jj] != 1:
                        #         # print("not equal to zero")
                        #         map_zero_one_hd[ii][jj] = 'X'
                    else:
                        path_temp.append(path[-1])
                        print(path_temp)
                        for rows in path_temp:
                            if rows == path_temp[-1]:
                                ii = rows[0] * 20 + 19
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)

                            elif rows == path_temp[0]:
                                ii = rows[0] * 20
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)

                            else:
                                ii = rows[0] * 20 + 10
                                jj = rows[1] * 20 + 10
                                temp = [ii, jj]
                                path_low_modulo2.append(temp)
                
        print("1over2 :", path_low_modulo2)
        
        height = 0
        for i in map_zero_one_hd:
            height +=1

        possible_number = [1, 2]

        for i in range(height):
            for j in range(len(map_zero_one_hd[0])):
                for index, path in enumerate(path_low):
                    if index == index_dij:
                        remplir = True
                        for rows in path:
                            if ((i < rows[0] * 20) or (i > rows[0] * 20 + 20) or (j < rows[1] * 20) or (j > rows[1] * 20 + 20)) and (map_zero_one_hd[i][j] not in possible_number):
                                pass
                            else:
                                remplir = False
                                break
                        if remplir:
                            map_zero_one_hd[i][j] = 3
                            
        
        global_path_HDmap = []

        for i in range(len(path_low_modulo2)-1):
            start = path_low_modulo2[i]
            end   = path_low_modulo2[i+1]
            map_path, path = Astar_from_map(map_zero_one_hd, start, end)

            # print("\nPath of HD mapeuuuh is : ", path, "\n")
            global_path_HDmap.append(path)
        
        for path in global_path_HDmap:
            for coord in path:
                map_zero_one_hd[coord[0]][coord[1]] = 9
        
        plot_map(map_zero_one_hd)
        
        matrix = np.array(map_zero_one_hd)

        name = "result/map/result/map_" + str(map_number) + "_result_HD.txt"
        
        with open(name, 'w') as f:
            for item in matrix:
                for i in range(len(item)):
                    if i == len(map_path[0]) - 1:
                        f.write("%s " % item[i])
                        f.write("\n")
                    else:
                        f.write("%s " % item[i])
        # print("####### GLOBAL PATH #######")
        # # for e in global_path_HDmap:
        # #     print(e)
        # print(global_path_HDmap)
        # print("###########################")

        to_save_txt.append(global_path_HDmap)
    
    for val in to_save_txt:
        print(val)
    
    # matrix = np.array(to_save_txt)

    name = "result/path/list_path_result_HD.txt"
    
    with open(name, 'w') as f:
        for item in to_save_txt:
                f.write("%s\n" % item)

    # for e in global_path_HDmap:
    #     print(e)

if __name__ == "__main__":

    #region DIJKSTRA : return fastest nodes in big graph ['1','4','5','6']  

    # start = '1'
    # end = '9'
    # print("FROM DIJKSTRA FILE : ", main_dijkstra(start, end),"\n")

    # path_graph = main_dijkstra(start, end)

    #endregion

    #region A* computation : return path which is coordinates list of the choosen path

    # map_number = '1'
    # name = "result/map/map_" + map_number + "_L.txt"
    # low_map = LOW_map_from_txt(name)

    # map_zero_one = map_to_ZeroOne(low_map)

    # for row in map_zero_one:
    #     print(row)
    # print("\nNumber of rows : ", len(map_zero_one))
    # print("Number of columns : ", len(map_zero_one[0]), "\n")


    # # print(map_zero_one)

    # start = [2, 1]
    # end = [8, 10]
    # cost = 1


    # if map_zero_one[start[0]][start[1]] != 1:
    #     map_path, path = Astar_from_map(map_zero_one, start, end, cost)

    #     # Map with path written :
    #     for row in map_path:
    #         print(row)

    #     print("\nPath is : ", path)
    # else:
    #     print("Start isn't walkable")

    #endregion

    #region pathfinding between maps

    # "FAKE" Dijkstra's return:
    dijkstra_list = ['1', '2', '3']

    # low_pathfinding(dijkstra_list)
    HD_pathfinding(dijkstra_list)

    #endregion

    #region Read saved maps from result/map/result
    
    # map_number = 3

    # name_minimap = "result/map/result/map_" + str(map_number) + "_result_HD.txt"
    
    # low_map = LOW_map_from_txt(name_minimap)
    
    # # for a in map_zero_one:
    # #     print(a)

    # plot_map(low_map)

    #endregion