import numpy as np
import cv2

class mini_map():
    nombre_mini_map = int(0)

    def __init__(self, tableau, option):
        # 1=HD 2=LOW
        mini_map.nombre_mini_map += int(1)
        self.map_number = int(mini_map.nombre_mini_map)
        self.option = option
        self.tableau = tableau
        self.tableau_low = self.get_low_map()
        self.save_map(1)
        self.reset_link_file = True
        self.reset_link_file_LOW = True
     
    def save_map(self,option):	
        #TODO: transform map to int8 (0,1)
        name = "result/map/map_"+str(self.map_number)
        if option == 1:
            #1=HD
            name += "_HD.txt"
        else:
            #0=LOW
            name += "_L.txt"

        f = open(name,"w+")

        for i in list(range(self.tableau.shape[0])):
            for j in list(range(self.tableau.shape[1])):
                f.write(str(self.tableau[i,j])+" ")
            f.write("\n")

        f.close()

    def create_like_file(self):
        name = "result/link/link_"+str(self.map_number)
        if self.option == 1:
            #1=HD
            name += "_HD.txt"
        else:
            #0=LOW
            name += "_L.txt"

        f = open(name,"w")

        f.write("")
        f.close()

    def add_link(self, new_link, option):
        name = "result/link/link_"+str(self.map_number)
        if option == 1:
            #1=HD
            name += "_HD.txt"
        else:
            #0=LOW
            name += "_L.txt"

        f = None
        if option == 1:
            if self.reset_link_file == True:
                f = open(name,"w")
                self.reset_link_file = False
            else:
                f = open(name,"a")
        else:
            if self.reset_link_file_LOW == True:
                f = open(name,"w")
                self.reset_link_file_LOW = False
            else:
                f = open(name,"a")

        f.write(new_link+"\n")
        f.close()

    def add_node_to_graph(self, node, first):
        # node is a numpy array need to convert to str.
        node_str = ""
        for i in list(range(node.shape[0])):
            node_str += str(int(node[i]))+" "

        # file path.
        name = "result/graph/all_graph.txt"

        # if minimap 1 so reset file.
        f = None
 
        if first:
            f = open(name,"w")
        else:
            f = open(name,"a")

        f.write(node_str+"\n")
        f.close()

    def get_value(self, square_meter_area):
        # this function get a [20x20] from tableau and decide if is it empty or obstacle.
        
        # first approch, rate of full
        number_of_case = 20*20
        ratio = 0.9
        value = number_of_case*ratio
        somme = 0
        for i in list(range(square_meter_area.shape[0])):
            for j in list(range(square_meter_area.shape[1])): 
                if(square_meter_area[i,j] > 250):
                    somme+=1
        if somme > value:
            return 255
        else:
            return 0

        # second one more technique.

    def get_low_map(self):
        print("alors",self.tableau.shape,self.tableau.shape[0]/20,self.tableau.shape[1]/20)
        tableau_low = np.zeros((int(self.tableau.shape[0]/20),int(self.tableau.shape[1]/20)))

        for i in list(range(tableau_low.shape[0])):
            for j in list(range(tableau_low.shape[1])):
                square_meter_area = self.tableau[i*20:((i+1)*20),j*20:((j+1)*20)]
                tableau_low[i,j] = self.get_value(square_meter_area)
        
        cv2.namedWindow("minimap_LOW",0)

        cv2.imshow("minimap_LOW", tableau_low)

        # save LOW version of mini map.
        name = "result/map/map_"+str(self.map_number)
        name += "_L.txt"

        f = open(name,"w+")

        for i in list(range(tableau_low.shape[0])):
            for j in list(range(tableau_low.shape[1])):
                f.write(str(int(tableau_low[i,j]))+" ")
            f.write("\n")

        f.close()


        return tableau_low