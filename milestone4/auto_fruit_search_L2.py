# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco


# import utility functions
sys.path.insert(0, "util")
from util.pibot import Alphabot
import util.measure as measure
import shutil

# import operate.py 
from operate import Operate
# import dijkstra.py
import dijkstra
import matplotlib.pyplot as plt
show_animation = True


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose, operate):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    
    wheel_vel = 10 # tick to move the robot
    # Extract x-coordinate and y-coordinate of the waypoint
    x_waypoint, y_waypoint = waypoint
    # Extract current x-coordinate, y-coordinate and orientation of the robot from robot_pose 
    x, y, theta = robot_pose
    # Compute horizontal distance between x-coordinate of the waypoint and the current x-coordinate of the robot	
    x_diff = x_waypoint - x
    # Compute vertical distance between y-coordinate of the waypoint and the current y-coordinate of the robot	
    y_diff = y_waypoint - y
    # Compute Euclidean distance between the robot and the waypoint
    distance_between_waypoint_and_robot = np.hypot(x_diff, y_diff)
    # Compute turning angle required by the robot to face the waypoint
    turning_angle =  np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])  #angle to goal
    print("Turning angle :{}".format(turning_angle))
    turn_time = ((abs(turning_angle)*baseline) / (2*wheel_vel*scale))
    diff_turn = turning_angle[0] - robot_pose[2][0]
    operate.take_pic()
    print("diff_turn: {:.2f}", diff_turn)
    print("Turning for {:.2f} seconds".format(turn_time))
        
    # turn towards the waypoint 
    if (turning_angle > 0): # rotating anticlockwise
        # Update robot_pose using SLAM 
        lv, rv = ppi.pibot.set_velocity(command, tick, turning_tick, run_time+0.0125)
        

        


        
    else: # rotating clockwise
        turn_time = ((abs(turning_angle)*baseline) / (2*wheel_vel*scale))
        print("Turning for {:.2f} seconds".format(turn_time)) 
        # Update robot_pose using SLAM 
        operator.take_pic()
        drive_meas = operator.control(operator_args, command=[0, -1], turning_tick=wheel_vel, run_time=turn_time, clockwise=1)
        operator.update_slam(drive_meas)
        
    wheel_vel = 25
    # after turning, drive straight to the waypoint
    drive_time = distance_between_waypoint_and_robot / (wheel_vel * scale) 
    print("Driving for {:.2f} seconds".format(drive_time))
    # Update robot_pose using SLAM 
    operator.take_pic()
    drive_meas = operator.control(operator_args, command=[1, 0], tick=wheel_vel, run_time=drive_time)
    operator.update_slam(drive_meas)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    # update the robot pose [x,y,theta]
    x_new = operator.ekf.robot.state[0,0]
    y_new = operator.ekf.robot.state[1,0]
    theta_new = operator.ekf.robot.state[2,0]
    robot_pose = [x_new, y_new, theta_new]  
    ####################################################

    return robot_pose

def drive_to_goal(robot_pose, rx, ry):
    for i in range(len(rx)):
        x = rx[len(rx)-i-1]
        y = ry[len(rx)-i-1]
        
        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        
        # estimate the robot's pose
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        ppi.set_velocity([0, 0])
    
# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    print(args)

    ppi = Alphabot(args.ip,args.port)
   
    operate = Operate(args)

    # Run SLAM 
    n_observed_markers = len(operator.ekf.taglist)
    if n_observed_markers == 0:
        if not operator.ekf_on:
            operator.notification = 'SLAM is running'
            operator.ekf_on = True
        else:
            operator.notification = '> 2 landmarks is required for pausing'
    elif n_observed_markers < 3:
        operator.notification = '> 2 landmarks is required for pausing'
    else:
        if not operator.ekf_on:
            operator.request_recover_robot = True
        operator.ekf_on = not operator.ekf_on
        if operator.ekf_on:
            operator.notification = 'SLAM is running'
        else:
            operator.notification = 'SLAM is paused'
            
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    
    # Obstacles positions
    ox, oy = [], []
    # Grid resolution 
    resolution = 0.4
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    robot_radius = baseline / 2
    robot_pose = get_robot_pose()
    
    # For loop to store x-coordinate and y-coordinate of aruco markers into ox and oy respectively
    for i in range(len(aruco_true_pos)):
        aruco_x = aruco_true_pos[i, :][0]
        aruco_y = aruco_true_pos[i, :][1]
        ox.append(aruco_x)
        oy.append(aruco_y)
    
    # For loop to store x-coordinate and y-coordinate of arena boundaries into ox and oy respectively
    for i in range(-160, 160): # Upper boundary 
        ox.append(i/100)
        oy.append(1.6)
    for i in range(-160, 160): # Lower Boundary
        ox.append(i/100)
        oy.append(-1.6)
    for i in range(-160, 160): # Left Boundary
        ox.append(-1.6)
        oy.append(i/100)
    for i in range(-160, 160): # Right Boundary 
        ox.append(1.6)
        oy.append(i/100)
    
    # For loop to loop through each fruit in search list 
    for i in range(len(search_list)):
        robot_pose = get_robot_pose()
        sx = robot_pose[0]
        sy = robot_pose[1]
        gx = fruits_true_pos[i, :][0]
        gy = fruits_true_pos[i, :][1]
        
        for j in range(len(fruits_true_pos)):
            if ((fruits_true_pos[j, :][0] != gx) and (fruits_true_pos[j, :][0] != gy)):
                ox.append(fruits_true_pos[j, :][0])
                oy.append(fruits_true_pos[j, :][1])
                
        if show_animation:  # pragma: no cover
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])    
            plt.xlabel("X"); plt.ylabel("Y")
            plt.xticks(space); plt.yticks(space)
            
        dijk = dijkstra.Dijkstra(ox, oy, resolution, robot_radius)
        rx, ry = dijk.planning(sx, sy, gx, gy)
        
        rx.pop(-1)
        ry.pop(-1)
        rx[0] = rx[0] + 0.2
        ry[0] = ry[0] - 0.2
        print(rx)
        print(ry)
        
        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.01)
            plt.show()
        fig = plt.figure 
        drive_to_goal(robot_pose, rx, ry)
        print("Reached {}".format(fruits_list[i]))
        ox.pop(-1)
        ox.pop(-1)
        oy.pop(-1)
        oy.pop(-1)
        time.sleep(3)
        
            
            
        
    
    
    
    
    
    