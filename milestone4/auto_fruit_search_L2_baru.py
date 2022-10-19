# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time
import pygame

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
from dijkstra import Dijkstra
#import a_star
sys.path.insert(0, "{}/autonomous".format(os.getcwd()))
from a_star import AStarPlanner
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
    fruit_true_pos_list = []
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
                fruit_true_pos_list.append([fruit_true_pos[i][0], fruit_true_pos[i][1]])
        n_fruit += 1
    
    print('fruits list:\n\n',fruit_true_pos_list)

    return(fruit_true_pos_list)

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, operate):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    
    threshold = 0.25
    error_counter = 999
    target_x = waypoint[0]
    target_y = waypoint[1]
    
    while error_counter > threshold:
        drive_time, turn_time = moving_param(operate, waypoint, operate.ekf.robot.state)

        #turning
        driving(operate, 1, turn_time)
        #go straight
        driving(operate, 0, drive_time)

        robot_x = operate.ekf.robot.state[0][0]
        robot_y = operate.ekf.robot.state[1][0]
        error = np.sqrt((robot_x - target_x)**2 + (robot_y - target_y)**2)
        if (error > threshold):
            get_robot_state(operate)
            robot_x = operate.ekf.robot.state[0][0]
            robot_y = operate.ekf.robot.state[1][0]
            error = np.sqrt((robot_x - target_x)**2 + (robot_y - target_y)**2)
        print("Current error:", error)
    ####################################################

    print("robot state:", operate.ekf.robot.state)
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

def get_robot_state(operate):
    driving(operate, -1 ,0) #update driving measurements

    turn_count = 12 #adjust based on the robot
    turn_angle = np.pi/2 #turn 90 degree
    vel_ang = operate.scale * operate.wheel_vel /(operate.baseline/2)
    turn_time = -turn_angle/turn_count/vel_ang

    start_angle = abs(operate.ekf.robot.state[2][0])
    new_angle = 0
    while new_angle < turn_angle:
        driving(operate,1, turn_time) #adjusting turning angle
        new_angle = abs(operate.ekf.robot.state[2][0]) - start_angle

    print("new angle = ", new_angle)

def moving_param(opreate, waypoint, robot_pose):
        baseline = operate.baseline
        scale = operate.scale

        goal_x, goal_y = waypoint[0], waypoint[1]
        robot_x, robot_y = robot_pose[0][0], robot_pose[1][0]

        dist_x = goal_x - robot_x
        dist_y = goal_y - robot_y

        robot_theta = get_robot_angle(operate)

        theta_turn_pos = 0
        theta_turn_neg = 0

        if dist_x == 0 and dist_y == 0:
            theta_turn = 0
        elif dist_x == 0:
            if dist_y > 0:
                theta_turn = np.pi/2
            else:
                theta_turn = 3*np.pi/2
        elif dist_y == 0:
            if dist_x > 0:
                theta_turn = 0
            else:
                theta_turn = np.pi
        else:
            if dist_y > 0 and dist_x > 0:
                theta_turn = abs(np.arctan(dist_y/dist_x))
            elif dist_y > 0 and dist_x < 0:
                theta_turn = np.pi - abs(np.arctan(dist_y/dist_x))
            elif dist_y < 0 and dist_x < 0:
                theta_turn = np.pi + abs(np.arctan(dist_y/dist_x))
            elif dist_y < 0 and dist_x > 0:
                theta_turn = np.pi*2 - abs(np.arctan(dist_y/dist_x))

        theta_turn_pos = theta_turn - robot_theta
        theta_turn_neg = (np.pi * 2 - abs(theta_turn_pos)) * -np.sign(theta_turn_pos)

        if abs(theta_turn_pos) < abs(theta_turn_neg):
            theta_turn = theta_turn_pos #can be pos or neg
        else:
            theta_turn = theta_turn_neg

        lin_vel = scale*operate.wheel_vel # m/tick * tick/s = m/s
        ang_vel = scale*operate.wheel_ang_vel/(baseline/2)
        turn_time = theta_turn/ang_vel

        dist = np.sqrt(dist_x**2 + dist_y**2) #distance in m
        drive_time = dist/lin_vel
        # print("From calc_move_param: theta_turn: ", theta_turn)
        return drive_time, turn_time


def get_robot_angle(operate):
    theta = abs(operate.ekf.robot.state[2][0]) % (np.pi *2)
    # print("From get_robot_theta: ekf: ", operate.ekf.robot.state[2][0])
    # print("From get_robot_theta: theta: ", theta)
    if (np.sign(operate.ekf.robot.state[2][0]) < 0):
        theta = np.pi*2 - theta

    return theta


def driving(operate, mode, time):
    drive_factor = 1.15 #change according to the robot condition
    turn_factor = 0
    if (mode == 1): #turning
        threshold_30 = np.pi/6/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))
        threshold_60 = np.pi/3/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))
        threshold_90 = np.pi/2/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))
        threshold_120 = np.pi/1.5/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))
        threshold_150 = np.pi/1.2/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))
        threshold_180 = np.pi/1/(operate.scale*operate.wheel_ang_vel/(operate.baseline/2))

        #adjust the factor according to the robot
        if (abs(time) <= threshold_30):
            turn_factor = 2.0
        elif (abs(time) > threshold_30 and abs(time) <= threshold_60):
            turn_factor = 1.90
        elif (abs(time) > threshold_60 and abs(time) <= threshold_90):
            turn_factor = 1.55
        elif (abs(time) > threshold_90 and abs(time) <= threshold_120):
            turn_factor = 1.30
        elif (abs(time) > threshold_120 and abs(time) <= threshold_150):
            turn_factor = 1.25
        elif (abs(time) > threshold_150 and abs(time) <= threshold_180):
            turn_factor = 1.19
        else:
            print("Error: turn angle more than 180 degree")

    direction = []
    if (mode == 0): #go straight
        direction = [1,0]
        robot_time = drive_factor * time
    elif (mode == 1): #turn
        direction = [0, np.sign(time)]
        robot_time = turn_factor * time
    else:
        direction = [0,0] #stay still
    
    robot_time = abs(robot_time)
    time = abs(time)

    lv, rv = operate.pibot.set_velocity(direction, operate.wheel_vel, operate.wheel_ang_vel, robot_time)
    operate.pibot.set_velocity([0, 0],0, 0, 0.25)

    operate.take_pic()
    drive_meas = measure.Drive(lv, rv, time)
    operate.update_slam(drive_meas)

    operate.draw(canvas)
    pygame.display.update()

    operate.take_pic()
    for i in range(15):
        drive_meas = measure.Drive(0, 0, 0)
        operate.update_slam(drive_meas)

        operate.draw(canvas)
        pygame.display.update()

def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    # update the robot pose [x,y,theta]
    robot_pose = [0.0, 0.0, 0.0]
    robot_pose = operate.ekf.robot.state  
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

def generate_space(fruits_true_pose, aruco_true_pose, search_index, fruits):
    ori_x, ori_y = [],[]

    #obstacle location
    for i in range(fruits):
        if i == search_index:
            continue
        ori_x.append(fruits_true_pose[i][0])
        ori_y.append(fruits_true_pose[i][1])

    for i in range(10):
        ori_x.append(aruco_true_pose[i][0])
        ori_y.append(aruco_true_pose[i][1])

    print("Number of obstacles: ", list(ori_x))

    # show the space map
    plt.plot(ori_x, ori_y, ".k")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.axis("equal")
    plt.show() 
    
    return ori_x, ori_y

def init_astar(obstacle):

    # start and goal position
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    grid_size = 0.2  # [m]
    grid_min = -1.6
    grid_max = 1.6
    robot_radius = 0.28  # [m]

    # set boundary
    ox, oy = [], []
    og = 0.1
    for i in np.arange(grid_min, grid_max+og, og):
        ox.append(i)
        oy.append(grid_min)
    for i in np.arange(grid_min, grid_max+og, og):
        ox.append(i)
        oy.append(grid_max)
    for i in np.arange(grid_min+og, grid_max, og):
        ox.append(grid_min)
        oy.append(i)
    for i in np.arange(grid_min+og, grid_max, og):
        ox.append(grid_max)
        oy.append(i)

    # set obstacle
    for point in obstacle:
        ox.append(point[0])
        oy.append(point[1])

    fig,axes=plt.subplots(nrows=1,ncols=1)

    major_ticks = np.arange(grid_min, grid_max+grid_size, grid_size)
    axes.set_xticks(major_ticks)
    axes.set_yticks(major_ticks)
    axes.grid()
    axes.set_title("Robot path")

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    # plt.plot(gx, gy, "xb")
    plt.xlim(grid_min-grid_size, grid_max+grid_size)
    plt.ylim(grid_min-grid_size, grid_max+grid_size)
    plt.show()

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

    return a_star

def path(rx, ry, target_x, target_y):
    grid_size = 0.2  # [m]
    grid_min = -1.6
    grid_max = 1.6

    fig,axes=plt.subplots(nrows=1,ncols=1)
    major_ticks = np.arange(grid_min, grid_max+grid_size, grid_size*2)
    minor_ticks = np.arange(grid_min, grid_max+grid_size, grid_size)

    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks)
    axes.set_yticks(major_ticks)


    plt.xlim(grid_min-grid_size, grid_max+grid_size)
    plt.ylim(grid_min-grid_size, grid_max+grid_size)
    axes.grid()
    axes.grid(which="minor",alpha=0.5)

    plt.plot(rx, ry, "-r")
    plt.plot(rx, ry, "xr")
    plt.plot(operate.ekf.robot.state[0][0], operate.ekf.robot.state[1][0], "og")
    plt.plot(target_x, target_y, "xb")
    plt.show()

# def plan_route(search_index):
#     fileB = "calibration/param/baseline.txt"
#     robot_radius = np.loadtxt(fileB, delimiter=',')*2   # robot radius = baseline of the robot/2.0
#     print(robot_radius)

#     print("Searching order:", search_index)
#     sx,sy = float(robot_pose[0]),float(robot_pose[1]) # starting location
#     gx,gy = fruits_true_pos[search_index][0],fruits_true_pos[search_index][1] # goal position

#     print("starting loation is: ",sx,",",sy)
#     print("ending loation is: ",gx,",",gy)
    
# #--------------------------------------- Using Dijkstra-------------------------------------#
#     if True:  # pragma: no cover
#         plt.plot(ori_x, ori_y, ".k")
#         plt.plot(sx, sy, "og")
#         plt.plot(gx, gy, "xb")
#         plt.grid(True)
#         plt.axis("equal")

#     grid_size = 0.2
#     dijkstra = Dijkstra(ori_x, ori_y, grid_size, robot_radius)
#     rx, ry = dijkstra.planning(sx, sy, gx, gy)
    
#     print("The x path is:",rx)
#     print("The y path is:",ry)
#     print("The last location is:",rx[-1])

#     if True:  # pragma: no cover
#         plt.plot(rx, ry, "-r")
#         plt.pause(0.01)
#         plt.show()
#     return rx,ry

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
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            operate.notification = 'SLAM is running'
            operate.ekf_on = True
        else:
            operate.notification = '> 2 landmarks is required for pausing'
    elif n_observed_markers < 3:
        operate.notification = '> 2 landmarks is required for pausing'
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            operate.notification = 'SLAM is running'
        else:
            operate.notification = 'SLAM is paused'

    # imports camera / wheel calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
            
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print(search_list)
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)


    fruits = 3 # number of fruits
    x,y = 0.0, 0.0
    robot_pose = [0.0, 0.0, 0.0]

    # pygame
    pygame.font.init()
    operate.TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    operate.TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 1020, 660
    canvas = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2
    
    operate.init_markers(aruco_true_pos)
    operate.ekf_on = True
    operate.scale = scale
    operate.baseline = baseline

    a_star = init_astar(np.concatenate((fruits_true_pos, aruco_true_pos), axis=0)) #generating space


    go = 1
    while go == 1:
        for target_pos in fruits_true_pos:
            robot_x = operate.ekf.robot.state[0][0]
            robot_y = operate.ekf.robot.state[1][0]
            offset_tx = [-0.2, -0.2, 0.2, 0.2]
            offset_ty = [0.2, -0.2, -0.2, 0.2]
            x_target = target_pos[0]
            y_target = target_pos[1]

            dist = []
            #finding location of the fruits
            for i in range(len(offset_tx)):
                dist.append(np.sqrt((offset_tx[i]+x_target-robot_x)**2 + (offset_ty[i]+y_target-robot_y)**2))

            min_index = dist.index(min(dist)) #nearest fruit

            x_target = x_target + offset_tx[min_index]
            y_target = y_target + offset_ty[min_index]
            ori_x, ori_y = generate_space(fruits_true_pos, aruco_true_pos, min_index, fruits)

            target_error = 99999
            thres = 0.35
            while target_error > thres:
                robot_x = operate.ekf.robot.state[0][0]
                robot_y = operate.ekf.robot.state[1][0]

                px, py = a_star.planning(robot_x, robot_y, x_target, y_target)
                px = np.flipud(px)
                py = np.flipud(py)
                px = np.delete(px, 0)
                py = np.delete(py, 0)

                path(px, py, x_target, y_target)

                for i in range(len(px)):
                    drive_time, turn_time = moving_param(operate, [px[i], py[i]], operate.ekf.robot.state)
                    drive_to_point([px[i], py[i]],operate)
                    # plot_path(px, py, x_target, y_target)

                get_robot_state(operate)
                robot_x = operate.ekf.robot.state[0][0]
                robot_y = operate.ekf.robot.state[1][0]
                target_error = np.sqrt((x_target-robot_x)**2 + (y_target-robot_y)**2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                go = 0
            pass

        go = 0



        
            
            
        
    
    
    
    
    
    