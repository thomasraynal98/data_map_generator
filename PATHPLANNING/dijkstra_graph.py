from collections import defaultdict
import os

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def edges_from_txt(path : str) -> list:

    #region edges[] exemple:

    # edges = [
    #     ('A', 'B', 5),
    #     ('B', 'C', 3),
    #     ('C', 'D', 2),
    #     ('B', 'D', 8),
    #     ('D', 'F', 7),
    #     ('D', 'E', 5),
    #     ('E', 'G', 3),
    #     ('F', 'G', 3),
    # ]

    #endregion

    name = "result/graph/all_graph.txt"
    f = open(name,"r")
    lines = f.readlines()
    f.close()

    # counter = 0
    # edges = []

    # for line in lines:
    #     counter += 1
    #     for i in range(len(line)):
    #         if line[i] != '0' and line[i] != ' ' and line[i] != '\n':
    #             edges.append((str(counter), line[i], 0))

    edges = [ (str(index + 1), line[i], 0) for index, line in enumerate(lines) for i, ele in enumerate(line) if line[i] != '0' and line[i] != ' ' and line[i] != '\n']


    # edges_removed = []
    # for i in range(len(edges)):
    #     for j in range(i,len(edges)):
    #         if edges[i][0] == edges[j][1]:
    #             if edges[i][1] == edges[j][0]:
    #                 edges_removed.append(edges[j])
    
    edges_removed = [ edges[j] for i, ele in enumerate(edges) for j in range(i, len(edges)) if edges[i][0] == edges[j][1] and edges[i][1] == edges[j][0] ]

    mixte_edge = [x for x in edges if x not in edges_removed]

    return mixte_edge

def add_weight_to_edges() -> list:

    # get values in all files in poid_graph folder

    filename_list = []

    directory = os.fsencode("result/poid_graph")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            filename_list.append(filename)
            continue
        else:
            continue

    filename_list = sorted(filename_list)
    lines_list = []

    for path in filename_list:
        path = "result/poid_graph/" + path
        with open(path, 'r') as f:
            lines_list.append(f.readlines())


    # Delete rows if empty list (rows = maps number; column = connections)

    # for elem in lines_list:
    #     if not elem:
    #         lines_list.remove(elem)
    
    lines_list = [ elem for elem in lines_list if elem ]


    clean_lines = []
    toto = []


    for i in range(len(lines_list)):
        for j in range(len(lines_list[i])):
            test = lines_list[i][j].split(' ')
            test = test[:-1]
            clean_lines.append(test)
        toto.append(clean_lines)
        clean_lines = []

    # Toto shape : [map1[[connection],[conection]], map2[[conection],[conection]], etc]

    # Create same shape list as [[map 1[]], [map 2[]], [map 3[]], etc] without same values in connection like 2, 4 is the same as 4, 2
    toto_removed = []
    same_shape_removed = []

    for i in range(len(toto)):
        for e in range(len(toto[i])):
            for j in range(e, len(toto[i])):
                if toto[i][e][0] == toto[i][j][1]:
                    if toto[i][e][1] == toto[i][j][0]:
                        toto_removed.append(toto[i][j])
        same_shape_removed.append(toto_removed)
        toto_removed = []

    connection_edges = []
    counter = 0
    for i in range(len(toto)):
        mixte_toto = [x for x in toto[i] if x not in same_shape_removed[counter]]
        connection_edges.append(mixte_toto)
        counter += 1

    # Transforme to Tuple to mimic edges = [('A', 'B', 5),('B', 'C', 3)]

    counter = 1
    edges = []
    for x in connection_edges:
        for y in x:
            edges.append((str(counter), y[0], int(y[2])))
            edges.append((str(counter), y[1], int(y[2])))
        counter += 1

    # Divide WEIGHT by 2 because add a new connection : 2, 4, 100 by map1 -> 1, 2, 50 and 1, 4, 50
 
    for i in range(len(edges)):
        list_edge = list(edges[i])
        new_weight = list_edge[2] // 2
        list_edge[2] = new_weight
        edges[i] = tuple(list_edge)


    temp = []

    for i in range(len(edges)):
        for j in range(i, len(edges)):
            if edges[i][0] == edges[j][0] and edges[i][1] == edges[j][1]:
                if edges[i][2] != edges[j][2]:
                    if edges[j] not in temp:
                        temp.append(edges[j])

    mixte_temp = [x for x in edges if x not in temp]
    # mixte_temp : tableau avec les valeurs additionnés entre la grosse liste de edges et la liste de "doublon"
    # IL RESTE A ADDITIONNER LES POIDS ENTRE LES DEUX TABLEAUX ! 

    # ADDITION DE WEIGHT go to list again:
    list_of_temp = [list(elem) for elem in temp]

    list_of_mixte = [list(elem) for elem in mixte_temp]

    for i in range(len(list_of_mixte)):
        for j in range(len(list_of_temp)):
            if list_of_mixte[i][0] == list_of_temp[j][0] and list_of_mixte[i][1] == list_of_temp[j][1]:
                list_of_mixte[i][2] += list_of_temp[j][2]


    remove_this = []
    for i in range(len(list_of_mixte)):
        for j in range(i + 1, len(list_of_mixte)):
            if list_of_mixte[i][0] == list_of_mixte[j][1]:
                if list_of_mixte[i][1] == list_of_mixte[j][0]:
                    remove_this.append(list_of_mixte[j])
                    list_of_mixte[i][2] += list_of_mixte[j][2]

    last_fusion = [x for x in list_of_mixte if x not in remove_this]

    update_weight_mixte = [tuple(x) for x in last_fusion]

    return update_weight_mixte

def main_dijkstra(start : str, end : str) -> list:
    # Create graph and add weight to edges !
    # add_weight_to_edges()
    edges = add_weight_to_edges()

    graph = Graph()

    for edge in edges:
        graph.add_edge(*edge)

    return dijsktra(graph, start, end)
