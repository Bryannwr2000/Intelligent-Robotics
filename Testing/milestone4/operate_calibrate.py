# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os,sys
import time
import json
import ast
import argparse
import matplotlib.pyplot as plt
import matplotlib
import _tkinter

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
# from A_star import AStarPlanner

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
# from network.scripts.detector import Detector

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'output2': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        
        #if args.ckpt == "":
            #self.detector = None
            #self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        #else:
            #self.detector = Detector(args.ckpt, use_gpu=False)
            #self.network_vis = np.ones((240, 320,3))* 100
            
        # self.detector = Detector(None, use_gpu=False)
        self.network_vis = np.ones((240, 320,3))* 100
        
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    #wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    def true_aruco(self, aruco_true_pos):
        x = []
        y = []
        meas = []

        for i in range(aruco_true_pos.shape[0]):
            x = aruco_true_pos[i][0]
            y = aruco_true_pos[i][1]
            tag = i + 1
            lms = measure.Marker(np.array([[x] ,[y]]), tag, 0.0*np.eye(2))
            meas.append(lms)

        self.ekf.add_landmarks(meas)

    def read_true_map(self, fname):
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

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.yolo_detection(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False
        # custom function

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))


    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 2]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -2]
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    def drive_to_point(self, waypoint):

        max_err = 0.25
        error = 100
        waypoint_x = waypoint[0]
        waypoint_y = waypoint[1]
        while error > max_err:
            drive_time, turn_time = self.robot_to_goal_time(waypoint, self.ekf.robot.state)

            self.drive_mode(1, turn_time) # mode 1 to turn

            self.drive_mode(0, drive_time) # mode 0 to drive straight

            robot_x = self.ekf.robot.state[0][0]
            robot_y = self.ekf.robot.state[1][0]
            error = np.sqrt((robot_x - waypoint_x) ** 2 + (robot_y - waypoint_y) ** 2)
            if error > max_err:
                self.get_robot_state()
                robot_x = self.ekf.robot.state[0][0]
                robot_y = self.ekf.robot.state[1][0]
                error = np.sqrt((robot_x - waypoint_x) ** 2 + (robot_y - waypoint_y) ** 2)
            print("Error from point: ", error)

    def update_robot_state(self):  ## turn 360 deg at the start for n times to get aruco locations, update self pose
        for i in range(2):
            self.get_robot_state()

    def get_robot_state(self):

        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        wheel_ang_vel = 15

        self.drive_mode(-1, 0)

        turn_count = 12   #each turn 30deg
        ang_vel = scale * wheel_ang_vel / (baseline / 2)
        turn_time = -(2*np.pi/turn_count) / ang_vel

        start_angle = abs(self.ekf.robot.state[2][0])
        rotated_angle = 0
        while rotated_angle <= 2*np.pi:
            self.drive_mode(1, turn_time)
            rotated_angle = abs(self.ekf.robot.state[2][0]) - start_angle

    def drive_mode(self, mode, time):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        wheel_vel = 30
        wheel_ang_vel = 15
        drive_factor = 1.15
        turn_factor = 0
        if mode == 1:
            ang_vel = (scale * wheel_ang_vel) / (baseline / 2)
            turnangle_30 = (np.pi / 6) / ang_vel
            turnangle_60 = (np.pi / 3) / ang_vel
            turnangle_90 = np.pi / 2 / ang_vel
            turnangle_120 = np.pi / 1.5 / ang_vel
            turnangle_150 = np.pi / 1.2 / ang_vel
            turnangle_180 = np.pi / 1 / ang_vel

            if abs(time) <= turnangle_30:
                turn_factor = 1.7 #previous = 2.0
            elif (abs(time) > turnangle_30) and (abs(time) <= turnangle_60):
                turn_factor = 1.6 #previous = 1.9
            elif (abs(time) > turnangle_60) and (abs(time) <= turnangle_90):
                turn_factor = 1.55
            elif (abs(time) > turnangle_90) and (abs(time) <= turnangle_120):
                turn_factor = 1.30
            elif (abs(time) > turnangle_120) and (abs(time) <= turnangle_150):
                turn_factor = 1.25
            elif (abs(time) > turnangle_150) and (abs(time) <= turnangle_180):
                turn_factor = 1.19

        actual_time = time

        if mode == 0:
            direction = [1, 0]
            actual_time = actual_time * drive_factor
        elif mode == 1:
            direction = [0, np.sign(time)]
            actual_time = actual_time * turn_factor
        else:
            direction = [0, 0]
        actual_time = abs(actual_time)
        time = abs(time)

        lv, rv = self.pibot.set_velocity(direction, wheel_vel, wheel_ang_vel, actual_time)
        self.pibot.set_velocity([0, 0], 0, 0, 0.5)

        self.take_pic()
        drive_meas = measure.Drive(lv, rv, time)
        self.update_slam(drive_meas)
        self.draw(canvas)
        pygame.display.update()
        self.take_pic()

        #for i in range(15):
        #    drive_meas = measure.Drive(0, 0, 0)
        #    self.update_slam(drive_meas)

        #    self.draw(canvas)
        #    pygame.display.update()

    def robot_to_goal_time(self, waypoint, robot_pose):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        wheel_vel = 30
        wheel_ang_vel = 15

        x_goal = waypoint[0]
        y_goal = waypoint[1]
        x_robot = robot_pose[0][0]
        y_robot = robot_pose[1][0]

        x_diff = x_goal - x_robot
        y_diff = y_goal - y_robot

        robot_theta = self.get_robot_theta()

        if x_diff == 0 and y_diff == 0:
            alpha = 0
        elif x_diff == 0 and y_diff > 0:
            alpha = np.pi / 2
        elif x_diff == 0 and y_diff < 0:
            alpha = 3 * np.pi / 2
        elif y_diff == 0 and x_diff > 0:
            alpha = 0
        elif y_diff == 0 and x_diff < 0:
            alpha = np.pi
        elif (y_diff > 0) and (x_diff > 0):
            alpha = abs(np.arctan(y_diff / x_diff))
        elif (y_diff > 0) and (x_diff < 0):
            alpha = np.pi - abs(np.arctan(y_diff / x_diff))
        elif (y_diff < 0) and (x_diff < 0):
            alpha = np.pi + abs(np.arctan(y_diff / x_diff))
        elif (y_diff < 0) and (x_diff > 0):
            alpha = np.pi * 2 - abs(np.arctan(y_diff / x_diff))

        alpha_pos = alpha - robot_theta
        alpha_neg = (np.pi * 2 - abs(alpha_pos)) * -np.sign(alpha_pos)

        if abs(alpha_pos) < abs(alpha_neg):
            alpha = alpha_pos
        else:
            alpha = alpha_neg
        lin_vel = scale * wheel_vel  # m/tick * tick/s = m/s
        ang_vel = scale * wheel_ang_vel / (baseline / 2)
        turn_times = alpha / ang_vel

        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)  # distance in m
        drive_time = dist / lin_vel
        # print("From calc_move_param: theta_turn: ", theta_turn)
        return drive_time, turn_time

    def get_robot_theta(self):
        theta = abs(self.ekf.robot.state[2][0]) % (np.pi * 2)  # to make sure theta is always in range from 0 to 2pi
        if self.ekf.robot.state[2][0] < 0:
            theta = np.pi * 2 - theta

        return theta

    def drive_straight_calib(self):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        wheel_vel = 30
        lin_vel = scale * wheel_vel  # m/tick * tick/s = m/s
        drive_time = 0.2 / lin_vel
        print("drive_time {:.2f}: ", format(drive_time))
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to move: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[1, 0], tick=wheel_vel, time=new_time)
            drive_factor = new_time / drive_time
            print("drive_factor: {:.2f}: ", format(drive_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The drive_factor is : {:.2f}: ", format(drive_factor))
                break


    def turn_calib(self):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        wheel_ang_vel = 15
        ang_vel = scale * wheel_ang_vel / (baseline / 2)

        print("Tuning for 30 deg")
        turnangle_30 = (np.pi / 6) / ang_vel
        print(turnangle_30)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_30_factor = new_time / turnangle_30
            print("turnangle_30_factor: {:.2f}: ", format(turnangle_30_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_30_factor is : {:.2f}: ", format(turnangle_30_factor))
                break

        print("Tuning for 60 deg")
        turnangle_60 = (np.pi / 3) / ang_vel
        print(turnangle_60)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_60_factor = new_time / turnangle_60
            print("turnangle_60_factor: {:.2f}: ", format(turnangle_60_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_60_factor is : {:.2f}: ", format(turnangle_60_factor))
                break


        print("Tuning for 90 deg")
        turnangle_90 = np.pi / 2 / ang_vel
        print(turnangle_90)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_90_factor = new_time / turnangle_90
            print("turnangle_90_factor: {:.2f}: ", format(turnangle_90_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_90_factor is : {:.2f}: ", format(turnangle_90_factor))
                break

        print("Tuning for 120 deg")
        turnangle_120 = np.pi / 1.5 / ang_vel
        print(turnangle_120)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_120_factor = new_time / turnangle_120
            print("turnangle_120_factor: {:.2f}: ", format(turnangle_120_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_30_factor is : {:.2f}: ", format(turnangle_120_factor))
                break

        print("Tuning for 150 deg")
        turnangle_150 = np.pi / 1.2 / ang_vel
        print(turnangle_150)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_150_factor = new_time / turnangle_150
            print("turnangle_150_factor: {:.2f}: ", format(turnangle_150_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_150_factor is : {:.2f}: ", format(turnangle_150_factor))
                break


        print("Tuning for 180 deg")
        turnangle_180 = np.pi / ang_vel
        print(turnangle_180)
        while True:
            new_time = 0.0
            new_time = input("time taken for robot to turn: ")
            try:
                new_time = float(new_time)
            except ValueError:
                print("Please enter a number.")
            self.pibot.set_velocity([0, 0], 0, 0, 3)
            self.pibot.set_velocity(command=[0, -1], turning_tick=wheel_ang_vel, time=new_time)
            turnangle_180_factor = new_time / turnangle_180
            print("turnangle_180_factor: {:.2f}: ", format(turnangle_180_factor))
            self.pibot.set_velocity([0, 0], 0, 0, 1)
            uInput = input("Calibration completed?[y/n]")
            if uInput == 'y':
                print("The turnangle_180_factor is : {:.2f}: ", format(turnangle_180_factor))
                break



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.103')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/best.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)



    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
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

    operate = Operate(args)
    start = True
    while start:
        # operate.drive_straight_calib()

        uInput = input("Continue to calib turnangle? [y/n]")
        if uInput == 'y':
            operate.turn_calib()
            uInput = input("Continue to calib straight motion? [y/n]")
            if uInput == 'y':
                continue
            else:
                break
        elif uInput == 'n':
            break







