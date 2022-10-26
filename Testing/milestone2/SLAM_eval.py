# evaluate the map generated by SLAM against the true map
import ast
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def parse_groundtruth(fname : str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def match_aruco_points(aruco0 : dict, aruco1 : dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
    return keys, np.hstack(points0), np.hstack(points1)

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])


    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1/num_points * np.reshape(np.sum(points1, axis=1),(2,-1))
    mu2 = 1/num_points * np.reshape(np.sum(points2, axis=1),(2,-1))
    sig1sq = 1/num_points * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/num_points * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1,0],R[0,0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed =  R @ points + x
    return points_transformed

def apply_transform_pred(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed =  R @ points - x
    print("points transformed:{}".format(points_transformed))
    return points_transformed



def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)

def load_robot_pose(fname):
    with open(fname,'r') as map_file:
        map_attributes = json.load(map_file)
    robot_x = map_attributes["robot_x"]
    robot_y = map_attributes["robot_y"]
    robot_theta = map_attributes["robot_theta"]
    
    return robot_x, robot_y, robot_theta

def save_txt(markers, fname="marker_pose_transformed.txt"):
    base_dir = Path('./')
    d = {}
    for i in range(10):
        d['aruco' + str(i+1) + '_0'] = {'x': round(markers[0][i], 1), 'y':round(markers[1][i], 1)}
    map_attributes = d
    with open(base_dir/fname,'w') as map_file:
        json.dump(map_attributes, map_file, indent=2)

def save_raw(markers, fname="marker_pose_raw.txt"):
    base_dir = Path('./')
    d = {}
    for i in range(10):
        d['aruco' + str(i+1) + '_0'] = {'x': round(markers[0][i], 1), 'y':round(markers[1][i], 1)}
    map_attributes = d
    with open(base_dir/fname,'w') as map_file:
        json.dump(map_attributes, map_file, indent=2)

def slam_eval():
    robot_x, robot_y, robot_theta = load_robot_pose("robot_pose.txt")
    gt_robot_x, gt_robot_y, gt_robot_theta = load_robot_pose("actual_robot_pose.txt")
    print("robot_x:{}".format(robot_x))
    gt_aruco = parse_groundtruth('TRUEMAP.txt')
    us_aruco = parse_user_map('lab_output/slam.txt')
    

    taglist, us_vec, gt_vec = match_aruco_points(us_aruco, gt_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]

    us_vec = us_vec[:,idx]

    gt_vec = gt_vec[:, idx]
    theta, x = solve_umeyama2d(us_vec, gt_vec)
    print(np.array([[robot_x], [robot_y]]).shape)
    # theta_robot, x_robot = solve_umeyama2d(np.array([[robot_x], [robot_y]]), np.array([[gt_robot_x], [gt_robot_y]]))
    us_vec_aligned = apply_transform(theta, x, us_vec)
    us_vec_pred_aligned = apply_transform_pred(robot_theta-gt_robot_theta, [[robot_x-gt_robot_x],[robot_y-gt_robot_y]], us_vec)
    
    diff = gt_vec - us_vec_aligned
    rmse = compute_rmse(us_vec, gt_vec)
    rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)
    rmse_cus_aligned = compute_rmse(us_vec_pred_aligned, gt_vec)
    
    
    
    save_txt(us_vec_pred_aligned)
    save_raw(us_vec)
    print("robot pose:{}".format(load_robot_pose("robot_pose.txt")))
    print()
    print("The following parameters optimally transform the estimated points to the ground truth.")
    print("Rotation Angle: {}".format(theta))
    print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    print()
    print("Number of found markers: {}".format(len(taglist)))
    print("RMSE before alignment: {}".format(rmse))
    print("RMSE after alignment:  {}".format(rmse_aligned))
    print("RMSE after custom alignment:  {}".format(rmse_cus_aligned))
    
    print()
    print('%s %7s %9s %7s %11s %9s %7s %13s %10s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy', 'Aligned x', 'Aligned y'))
    print('---------------------------------------------------------------------------------------')
    for i in range(len(taglist)):
        print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], gt_vec[0][i], us_vec[0][i], diff[0][i], gt_vec[1][i], us_vec[1][i], diff[1][i], us_vec_pred_aligned[0][i], us_vec_pred_aligned[1][i]))
    
    ax = plt.gca()
    ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(us_vec[0,:], us_vec[1,:], marker='x', color='C1', s=100)
    ax.scatter(us_vec_aligned[0, :], us_vec_aligned[1, :], marker='^', color='C2', s=100)
    ax.scatter(us_vec_pred_aligned[0, :], us_vec_pred_aligned[1, :], marker='1', color='C3', s=100)
    for i in range(len(taglist)):
        ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
        ax.text(us_vec[0,i]+0.05, us_vec[1,i]+0.05, taglist[i], color='C1', size=12)
        ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C2', size=12)
        ax.text(us_vec_pred_aligned[0,i]+0.05, us_vec_pred_aligned[1,i]+0.05, taglist[i], color='C2', size=12)

    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Real','Predicted','Pred_aligned(TrueMap)', 'Pred_aligned(Trans)'])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    slam_eval()
    #import argparse

    #parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    #parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
    #parser.add_argument("estimate", type=str, help="The estimate file name.")
    #args = parser.parse_args()

    #gt_aruco = parse_groundtruth(args.groundtruth)
    #us_aruco = parse_user_map(args.estimate)

    #taglist, us_vec, gt_vec = match_aruco_points(us_aruco, gt_aruco)
    #idx = np.argsort(taglist)
    #taglist = np.array(taglist)[idx]
    #us_vec = us_vec[:,idx]
    #gt_vec = gt_vec[:, idx] 

    #theta, x = solve_umeyama2d(us_vec, gt_vec)
    #us_vec_aligned = apply_transform(theta, x, us_vec)
    
    #diff = gt_vec - us_vec_aligned
    #rmse = compute_rmse(us_vec, gt_vec)
    #rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)
    
    #print()
    #print("The following parameters optimally transform the estimated points to the ground truth.")
    #print("Rotation Angle: {}".format(theta))
    #print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    #print()
    #print("Number of found markers: {}".format(len(taglist)))
    #print("RMSE before alignment: {}".format(rmse)) 
    #print("RSME after transformed: {}".format(rsme_transformed))
    #print("RMSE after alignment:  {}".format(rmse_aligned)) #Alignment with true map
    
    #print()
    #print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    #print('-----------------------------------------------------------------')
    #for i in range(len(taglist)):
    #    print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], gt_vec[0][i], us_vec_aligned[0][i], diff[0][i], gt_vec[1][i], us_vec_aligned[1][i], diff[1][i]))
    
    #ax = plt.gca()
    #ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    #ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
    #for i in range(len(taglist)):
    #    ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
    #    ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
    #plt.title('Arena')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    #ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    #plt.legend(['Real','Pred'])
    #plt.grid()
    #plt.show()

    #d={}
    #for i in range(len(taglist)):
    #    d["aruco" + str(i+1) + "_0"] = {"x": us_vec_transformed[0][i], "y": us_vec_transformed[1][i]}

    #with open(base_dir/'PredMap.txt', 'w') as f:
    #    json.dump(d,f,indent =4)