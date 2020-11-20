# data_map_generator

This is a generator of data for RANE MK3 algortyhme, make in november 2020.
Put your image in image folder and get the path to your image in line 404.

# exemple 1 : Normal place.
![Alt text](/image/real2.jpg?raw=true "Schema data structure")

# data structure.
## folder map.
Name of file = "map_" + "number of map_" + "quality of map.txt"
STRUCTURE EXPLICATION
* Line correspond to line in np.array.
* Colone correspond to colone in np.array.
* value 0 correspond to empty area.
* value 255 correspond to full area.

## folder link.
Name of file = "link_" + "number of map_" + "quality of map.txt"
STRUCTURE EXPLICATION
* each line correspond to a connection. format (A B C D E)
* The number of map in title correspond to the number of current map.
* A = correspond to x coordinate in current map.
* B = correspond to y coordinate in current map.
* C = correspond to x coordinate in E map.
* D = correspond to y coordinate in E map.
* E = correspond to the map that are connected to current map.

## folder graph.
Name of only file = "all_graph.txt"
STRUCTURE EXPLICATION
* each line correspond to a node. format (X X ... X)
* the number of X value in each line correspond to the total of node in graph.
* each X value correspond to the number of node (map) that are connected to the currently node (correspond to the line number)
EXEMPLE
* this file:
** 2 0
** 1 0
EXPLICATION EXEMPLE
* there is 2 nodes in this graph (2 maps)
* the node 1 is represented by the line number 1.
* the node 2 is represented by the line number 2.
* the node 1 is connected to the node 2. (first line information)
* the node 2 is connected to the node 1. (seconde line information)

## folder poid_graph
Name of file = "weight_" + "number of map_" + "quality of map.txt"
INFORMATION: in algorthme we only need HD weight so there are only HD weight.
STRUCTURE EXPLICATION
* all file correspond to all weight on the map. format (B C D)
* B correspond to map B.
* C correspond to map C.
* D correspond to distance in pixel from map B to map C cross map A.
NUMBER OF WEIGHT
* for a map with 1 connection = 0 weight X 2 (X2 because there are doublon but the information it's the same)
because a weight correspond to the distance to parcour from map B to map C in the map A.
* for a map with 2 connection = 1 weight X 2 
* for a map with 3 connection = 3 weights X 2
* for a map with 4 connection = 6 weights X 2
* for a map with 5 connection = 10 weights X 2
* for a map with N connection = for(i=N-1,i >= 0, N-1) somme += i

# Visual explication

![Alt text](/image/126518612_838970940227245_7258546604859959701_n.jpg?raw=true "Schema data structure")
