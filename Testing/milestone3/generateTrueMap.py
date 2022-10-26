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
    true_map = open("FinalMap.txt","x")
    true_map_dict = {}
    arucos_file = open("PredMap.txt","r")
    arucos_position = arucos_file.read()
    arucos_dict = json.loads(arucos_position)
    true_map_dict.update(arucos_dict)
    
    fruits_file = open("lab_output/targets.txt","r")
    fruits_position = fruits_file.read()
    fruits_dict = json.loads(fruits_position)
    true_map_dict.update(fruits_dict)
    true_map.write(str(true_map_dict))
    