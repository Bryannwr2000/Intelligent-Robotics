import argparse
import json
import ast 
import numpy as np

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict
    
if __name__ == '__main__':
    true_map = open("M5_true_map.txt","x")
    aruco_dict = parse_user_map("lab_output/slam.txt")
    true_map_dict = {}
    aruco_num = ""
    for i in range(len(aruco_dict)):
        aruco_pos = aruco_dict[i+1]
        aruco_num = "aruco" + str(i+1) + "_" + "0"
        aruco_dict_1 = {"x'": aruco_pos[0][0]), "y": aruco_pos[1][0]}
        aruco_dict_2 = {aruco_num: aruco_dict_1}
        true_map_dict.update(aruco_dict_2)
    
    fruits_file = open("lab_output/targets.txt","r")
    fruits_position = fruits_file.read()
    fruits_dict = json.loads(fruits_position)
    true_map_dict.update(fruits_dict)
    true_map.write(str(true_map_dict))
    
