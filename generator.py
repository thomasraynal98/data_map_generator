import numpy as np
import cv2
import time
import math as m 

from mini_map import * 
#pute
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
                la_map[i,j] = EMPTY
            else:
                la_map[i,j] = OBSTACLE

    return la_map

ref_point = np.zeros(4,dtype=np.uint16)
cropping = False
first_rectangle = True
list_map = []
ma_map_global = np.zeros(1)
mask_map = np.zeros(1)

def process_post_selection():
    global first_rectangle, list_map

    if (ref_point[0] < ref_point[2]) and (ref_point[1] < ref_point[3]):
        cropmap = ma_map_global[ref_point[1]:ref_point[3],ref_point[0]:ref_point[2]]

    if (ref_point[0] < ref_point[2]) and (ref_point[1] > ref_point[3]):
        cropmap = ma_map_global[ref_point[3]:ref_point[1],ref_point[0]:ref_point[2]]

    if (ref_point[0] > ref_point[2]) and (ref_point[1] < ref_point[3]):
        cropmap = ma_map_global[ref_point[1]:ref_point[3],ref_point[2]:ref_point[0]]

    if (ref_point[0] > ref_point[2]) and (ref_point[1] > ref_point[3]):
        cropmap = ma_map_global[ref_point[3]:ref_point[1],ref_point[2]:ref_point[0]]

    mini_mapX = mini_map(cropmap,1)
    first_rectangle = False

    complete_mask_map2(mini_mapX)
    recompute_node_file(mini_mapX)
    compute_weight()
    list_map.append(mini_mapX)

    cv2.imshow("minimap", cropmap)

def recompute_node_file(mini_mapY):
    node_graph = np.zeros((mini_map.nombre_mini_map,mini_map.nombre_mini_map)) # like a football score tab.

    # to solve problem.
    first = True
    for i in list(range(mini_map.nombre_mini_map)):
        # run into all link file.
        name = "result/link/link_"+str(i+1)
        if True:
            #1=HD
            name += "_HD.txt"
        else:
            #TODO
            #0=LOW
            name += "_L.txt"

        try:
            f = open(name,"r")

            # read all lines
            liste = np.zeros(mini_map.nombre_mini_map)
            for x in f:
                line = x.split((" "))
                liaison = line[4].split("\n")[0]
                if int(liaison) not in liste:
                    index = np.where(liste == 0)
                    liste[index[0][0]] = liaison      
            f.close()
            mini_mapY.add_node_to_graph(liste, first)

            # to solve problem.
            first = False

        except:
            pass
            no_connected_node = np.zeros(mini_map.nombre_mini_map)
            mini_mapY.add_node_to_graph(no_connected_node, first)

def compute_distance_dumb_method(x1, y1, x2, y2):
    #TODO: make version with A*
    tempo = (((int(x2)-int(x1))**2)+((int(y2)-int(y1))**2))**0.5
    return int(tempo)

def compute_weight():
    for i in list(range(mini_map.nombre_mini_map)):
        # run into all link file.
        name = "result/link/link_"+str(i+1)
        if True:
            #1=HD
            name += "_HD.txt"
        else:
            #TODO
            #0=LOW
            name += "_L.txt"
        
        f = open(name,"r")

        # our structure for mapAWeight. (mapB, mapC, weight)
        data_structure_weight = None
        contenue_file = None
        ligne_index = 0
        for x in f:
            line = x.split(" ")
            line[4] = line[4].split("\n")[0]
            if contenue_file is None:
                contenue_file = np.array([[line[0],line[1],line[2],line[3],line[4]]],dtype=np.uint32)
            else:
                contenue_file = np.concatenate((contenue_file,np.array([[line[0],line[1],line[2],line[3],line[4]]],dtype=np.uint32)),axis=0)
        f.close()

        if(contenue_file is not None):
            for line in contenue_file:
                # get coordonate of mapB point in mapA ref.
                coordonateB = np.array([line[0],line[1]])
                from_map = line[4]

                # now compare with all value in contenue_file.
                for line_match in contenue_file:
                    coordonateC = np.array([line_match[0],line_match[1]])
                    to_map = line_match[4]

                    if to_map != from_map:
                        # if we don't have datastructure create it.
                        distance = compute_distance_dumb_method(coordonateB[0],coordonateB[1],coordonateC[0],coordonateC[1])
                        if distance == 0:
                            print()
                        if data_structure_weight is None:
                            data_structure_weight = np.array([[from_map,to_map,distance]])
                        else:
                            # now found if we have already the path weight or if we need to create it.
                            r = np.where(data_structure_weight[:,0] == from_map)
                            if r[0].shape[0] != 0:
                                # we have path from mapB but maybe not to mapC.
                                # to solve bug.
                                found_it = False
                                for index in r[0]:
                                    if data_structure_weight[index,1] == to_map:
                                        # we have already the path.
                                        data_structure_weight[index, 2] = int((data_structure_weight[index, 2]+distance)/2)
                                        found_it = True
                                        break
                                if found_it == False:
                                    # we don't found this path, add new.
                                    data_structure_weight = np.concatenate((data_structure_weight,np.array([[from_map,to_map,distance]])),axis=0)
                            else:
                                data_structure_weight = np.concatenate((data_structure_weight,np.array([[from_map,to_map,distance]])),axis=0)
        
        if data_structure_weight is not None:
            # save all weight in his corresponding file.
            name = "result/poid_graph/weight_"+str(i+1)
            if True:
                #1=HD
                name += "_HD.txt"
            else:
                #TODO
                #0=LOW
                name += "_L.txt"
            
            f = open(name,"w")
            for lines in data_structure_weight:
                weight_data = ""
                for data in lines:
                    weight_data += str(data)+" "
                f.write(weight_data+"\n")
            f.close()
        else:
            # save nothing in his corresponding file, so create it.
            name = "result/poid_graph/weight_"+str(i+1)
            if True:
                #1=HD
                name += "_HD.txt"
            else:
                #TODO
                #0=LOW
                name += "_L.txt"
            
            f = open(name,"w")
            f.write("")
            f.close()

def complete_mask_map2(mini_mapX):   
    global mask_map, list_map, ref_point
    # draw frontiere on maskMap + (x, y) coordinates in minimap.
    # draw four side.
        
    # put ref_point in good direction      
    if ref_point[0] > ref_point[2]:
        tempo = ref_point[2]
        ref_point[2] = ref_point[0]
        ref_point[0] = tempo
    if ref_point[1] > ref_point[3]:
        tempo = ref_point[3]
        ref_point[3] = ref_point[1]
        ref_point[1] = tempo

    #test
    tempo1 = ref_point[0]
    ref_point[0] = ref_point[1]
    ref_point[1] = tempo1

    tempo1 = ref_point[2]
    ref_point[2] = ref_point[3]
    ref_point[3] = tempo1

    L = ref_point[3]-ref_point[1]
    H = ref_point[2]-ref_point[0]

    print("taille", ref_point, "hauteur", H, "largeur", L)
    mini_mapX.create_like_file() 

    # for LOW MAP.
    compteur_vertical_A = 0
    compteur_vertical_B = 0
    seuil = 18

    for h in list(range(H)):
        # for LOW MAP
        if h % 20 == 0:
            compteur_vertical_A = 0
            compteur_vertical_B = 0
        # on parcours les deux lignes vertical.macsk_map[ref_point[0],ref_point[1]+i
        # 1 ligne vertical gauche
        if (mask_map[ref_point[0]+h,ref_point[1],0] != 0) and (mask_map[ref_point[0]+h,ref_point[1],0] != mini_map.nombre_mini_map):
            print("cas1")
            # we need to check if this link is empty of full, if empty go link.
            if ma_map_global[ref_point[0]+h,ref_point[1]] == 255:
                print("cas1_link")
                link = str(h)+" "+str(0)+" "+str(mask_map[ref_point[0]+h,ref_point[1],1])+" "+str(mask_map[ref_point[0]+h,ref_point[1],2])+" "+str(mask_map[ref_point[0]+h,ref_point[1],0])
                mini_mapX.add_link(link,1)
                reverse_link = str(mask_map[ref_point[0]+h,ref_point[1],1])+" "+str(mask_map[ref_point[0]+h,ref_point[1],2])+" "+str(h)+" "+str(0)+" "+str(mini_mapX.map_number)
                list_map[mask_map[ref_point[0]+h,ref_point[1],0]-1].add_link(reverse_link,1)
            # for LOW MAP
                compteur_vertical_A += 1
                if compteur_vertical_A >= seuil:
                    compteur_vertical_A = 0
                    # create new link in LOW MAP.
                    x_coord_ref_low = int(h//20)
                    y_coord_ref_low = int(0//20)
                    xLINK_coord_ref_low = int(mask_map[ref_point[0]+h,ref_point[1],1]//20)
                    yLINK_coord_ref_low = int(mask_map[ref_point[0]+h,ref_point[1],2]//20)
                    to_map = int(mask_map[ref_point[0]+h,ref_point[1],0])
                    mini_mapX.add_link(str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(to_map)+" ",0)
                    list_map[mask_map[ref_point[0]+h,ref_point[1],0]-1].add_link(str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(mini_mapX.map_number),0)
                
        if (mask_map[ref_point[0]+h,ref_point[1],0] != mini_map.nombre_mini_map):
            mask_map[ref_point[0]+h,ref_point[1],0] = mini_map.nombre_mini_map
            mask_map[ref_point[0]+h,ref_point[1],2] = 0
            mask_map[ref_point[0]+h,ref_point[1],1] = h
        
        # 2 ligne vertical droite
        if (mask_map[ref_point[0]+h,ref_point[3],0] != 0) and (mask_map[ref_point[0]+h,ref_point[3],0] != mini_map.nombre_mini_map):
            print("cas2")
            # we need to check if this link is empty of full, if empty go link.
            if ma_map_global[ref_point[0]+h,ref_point[3]] == 255:
                print("cas2_link")
                link = str(h)+" "+str(L-1)+" "+str(mask_map[ref_point[0]+h,ref_point[3],1])+" "+str(mask_map[ref_point[0]+h,ref_point[3],2])+" "+str(mask_map[ref_point[0]+h,ref_point[3],0])
                mini_mapX.add_link(link,1)
                reverse_link = str(mask_map[ref_point[0]+h,ref_point[3],1])+" "+str(mask_map[ref_point[0]+h,ref_point[3],2])+" "+str(h)+" "+str(L-1)+" "+str(mini_mapX.map_number)
                list_map[mask_map[ref_point[0]+h,ref_point[3],0]-1].add_link(reverse_link,1)

                compteur_vertical_B += 1
                if compteur_vertical_B >= seuil:
                    compteur_vertical_B = 0
                    # create new link in LOW MAP.
                    x_coord_ref_low = int(h//20)
                    y_coord_ref_low = int((L-1)//20)
                    xLINK_coord_ref_low = int(mask_map[ref_point[0]+h,ref_point[3],1]//20)
                    yLINK_coord_ref_low = int(mask_map[ref_point[0]+h,ref_point[3],2]//20)
                    to_map = int(mask_map[ref_point[0]+h,ref_point[3],0])
                    mini_mapX.add_link(str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(to_map)+" ",0)
                    list_map[mask_map[ref_point[0]+h,ref_point[3],0]-1].add_link(str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(mini_mapX.map_number),0)

        if (mask_map[ref_point[0]+h,ref_point[3],0] != mini_map.nombre_mini_map):
            mask_map[ref_point[0]+h,ref_point[3],0] = mini_map.nombre_mini_map
            mask_map[ref_point[0]+h,ref_point[3],2] = L-1
            mask_map[ref_point[0]+h,ref_point[3],1] = h

    # for LOW MAP.
    compteur_horizontal_A = 0
    compteur_horizontal_B = 0

    for l in list(range(L)):
        if l % 20 == 0:
            compteur_horizontal_A = 0
            compteur_horizontal_B = 0
        # on parcours les deux lignes horizontal.
        # 3 ligne horizontal haut
        if (mask_map[ref_point[0],ref_point[1]+l,0] != 0) and (mask_map[ref_point[0],ref_point[1]+l,0] != mini_map.nombre_mini_map):
            print("cas3")
            # we need to check if this link is empty of full, if empty go link.
            if ma_map_global[ref_point[0],ref_point[1]+l] == 255:
                print("cas3_link")
                link = str(0)+" "+str(l)+" "+str(mask_map[ref_point[0],ref_point[1]+l,1])+" "+str(mask_map[ref_point[0],ref_point[1]+l,2])+" "+str(mask_map[ref_point[0],ref_point[1]+l,0])
                mini_mapX.add_link(link,1)
                reverse_link = str(mask_map[ref_point[0],ref_point[1]+l,1])+" "+str(mask_map[ref_point[0],ref_point[1]+l,2])+" "+str(0)+" "+str(l)+" "+str(mini_mapX.map_number)
                list_map[mask_map[ref_point[0],ref_point[1]+l,0]-1].add_link(reverse_link,1)

                compteur_horizontal_A += 1
                if compteur_horizontal_A >= seuil:
                    compteur_horizontal_A = 0
                    # create new link in LOW MAP.
                    x_coord_ref_low = int(0//20)
                    y_coord_ref_low = int(l//20)
                    xLINK_coord_ref_low = int(mask_map[ref_point[0],ref_point[1]+l,1]//20)
                    yLINK_coord_ref_low = int(mask_map[ref_point[0],ref_point[1]+l,2]//20)
                    to_map = int(mask_map[ref_point[0],ref_point[1]+l,0])
                    mini_mapX.add_link(str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(to_map)+" ",0)
                    list_map[mask_map[ref_point[0],ref_point[1]+l,0]-1].add_link(str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(mini_mapX.map_number),0)

        if (mask_map[ref_point[0],ref_point[1]+l,0] != mini_map.nombre_mini_map):
            mask_map[ref_point[0],ref_point[1]+l,0] = mini_map.nombre_mini_map
            mask_map[ref_point[0],ref_point[1]+l,2] = l
            mask_map[ref_point[0],ref_point[1]+l,1] = 0
        
        # 4 ligne horizontal bas
        if (mask_map[ref_point[2],ref_point[1]+l,0] != 0) and (mask_map[ref_point[2],ref_point[1]+l,0] != mini_map.nombre_mini_map):
            print("cas4")
            # we need to check if this link is empty of full, if empty go link.
            if ma_map_global[ref_point[2],ref_point[1]+l] == 255:
                print("cas4_link")
                link = str(H-1)+" "+str(l)+" "+str(mask_map[ref_point[2],ref_point[1]+l,1])+" "+str(mask_map[ref_point[2],ref_point[1]+l,2])+" "+str(mask_map[ref_point[2],ref_point[1]+l,0])
                mini_mapX.add_link(link,1)
                reverse_link = str(mask_map[ref_point[2],ref_point[1]+l,1])+" "+str(mask_map[ref_point[2],ref_point[1]+l,2])+" "+str(H-1)+" "+str(l)+" "+str(mini_mapX.map_number)
                list_map[mask_map[ref_point[2],ref_point[1]+l,0]-1].add_link(reverse_link,1)

                compteur_horizontal_B += 1
                if compteur_horizontal_B >= seuil:
                    compteur_horizontal_B = 0
                    # create new link in LOW MAP.
                    x_coord_ref_low = int((H-1)//20)
                    y_coord_ref_low = int(l//20)
                    xLINK_coord_ref_low = int(mask_map[ref_point[2],ref_point[1]+l,1]//20)
                    yLINK_coord_ref_low = int(mask_map[ref_point[2],ref_point[1]+l,2]//20)
                    to_map = int(mask_map[ref_point[2],ref_point[1]+l,0])
                    mini_mapX.add_link(str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(to_map)+" ",0)
                    list_map[mask_map[ref_point[2],ref_point[1]+l,0]-1].add_link(str(xLINK_coord_ref_low)+" "+str(yLINK_coord_ref_low)+" "+str(x_coord_ref_low)+" "+str(y_coord_ref_low)+" "+str(mini_mapX.map_number),0)

            
        if (mask_map[ref_point[2],ref_point[1]+l,0] != mini_map.nombre_mini_map):
            mask_map[ref_point[2],ref_point[1]+l,0] = mini_map.nombre_mini_map
            mask_map[ref_point[2],ref_point[1]+l,2] = l
            mask_map[ref_point[2],ref_point[1]+l,1] = H-1

def rectangle_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, cropping, ma_map_global, list_angle_detector

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_MBUTTONDOWN:
        if int(x) % 20 == 0:
            ref_point[0] = int(x)
        else:
            if int(x) % 20 > 10:
                ref_point[0] = int(x) + (20-(int(x) % 20))
            else:
                ref_point[0] = int(x) - (int(x) % 20)

        if int(y) % 20 == 0:
            ref_point[1] = int(y)
        else:
            if int(y) % 20 > 10:
                ref_point[1] = int(y) + (20-int(y) % 20)
            else:
                ref_point[1] = int(y) - (int(y) % 20)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_MBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if int(x) % 20 == 0:
            ref_point[2] = int(x)
        else:
            if int(x) % 20 > 10:
                ref_point[2] = int(x) + (20-(int(x) % 20))
            else:
                ref_point[2] = int(x) - (int(x) % 20)

        if int(y) % 20 == 0:
            ref_point[3] = int(y)
        else:
            if int(y) % 20 > 10:
                ref_point[3] = int(y) + (20-int(y) % 20)
            else:
                ref_point[3] = int(y) - (int(y) % 20)

        # draw a rectangle around the region of interest
        if (ma_map_affichage[ref_point[1],ref_point[0]] == 150) or first_rectangle:
            print("yaaaa",ref_point)
            cv2.rectangle(ma_map_affichage, (ref_point[0],ref_point[1]), (ref_point[2],ref_point[3]), (150,0,0), 1)
            cv2.imshow("fenetre", ma_map_affichage)
            process_post_selection()

def clean_graph_file():
    name = "result/graph/all_graph.txt"
    f = open(name,"w")
    f.write("")
    f.close()

if __name__ == "__main__":
    clean_graph_file()
    ma_map = import_image_and_transform("image/real3.jpg")
    ma_map_global = ma_map
    ma_map_affichage = ma_map.copy()
    mask_map = np.zeros((ma_map_global.shape[0], ma_map_global.shape[1], 3),dtype=np.uint16)

    print(cv2.WINDOW_NORMAL)
    cv2.namedWindow("fenetre",0)
    cv2.imshow("fenetre", ma_map_affichage)
    cv2.setMouseCallback("fenetre", rectangle_selection)

    while True:
        # display the image and wait for a keypress
        cv2.imshow("fenetre", ma_map_affichage)
        key = cv2.waitKey(1) & 0xFF

        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break

    cv2.destroyAllWindows()