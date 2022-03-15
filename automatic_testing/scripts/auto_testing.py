import math
from os import kill
import subprocess
import sys

import rosdep2
from kill_functions import *
import matplotlib.pyplot as plt
import time
import gazebo_msgs.srv
import rospy
kill_env()

for n_intersections in [2]:
    obs = [2]
    for n_obstacles_per_tile in obs:
        if sys.argv.__len__() > 2:
            n_tests = sys.argv[-1]
        else:
            n_tests = 10
        for _ in range(n_tests):
            test_number = str(n_intersections)+"_" +str(n_obstacles_per_tile) + "_" + str(n_tests)
            init_time = time.time()
            process = subprocess.Popen(
                ['python3', '/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/scripts/gen_w_and_launch_gazebo.py', 
                str(n_intersections),
                str(test_number),
                str(n_obstacles_per_tile)])
            time.sleep(2)
            rospy.init_node("testing_node")
            rospy.set_param("/test_status", "ongoing")

            failure_by_time = False
            while rospy.get_param("/test_status") == "ongoing":
                time_passed =  time.time() - init_time
                if time_passed > 60*10:
                    failure_by_time = True
                    break
                time.sleep(0.5)
            if failure_by_time:
                f.write("success || {} || {} || {}\n".format(name, n_intersections, n_obstacles_per_tile))
            else:
                x_obj = rospy.get_param("/x_obj")
                y_obj = rospy.get_param("/y_obj")
                name = rospy.get_param("/env_name")

                service_name = "/gazebo/get_model_state"
                rospy.wait_for_service(service_name)
                get_model_state = rospy.ServiceProxy(
                    service_name, gazebo_msgs.srv.GetModelState)
                p = get_model_state("/", "").pose.position

                distance = math.sqrt((p.x-x_obj)**2 + (p.y-y_obj)**2)
                with open("/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/scripts/results_{}.txt".format(test_number),"a") as f:
                    if distance < 10 and not failure_by_time:
                        f.write("success || {} || {} || {}\n".format(name, n_intersections, n_obstacles_per_tile))
                    else:
                        f.write("failure || {} || {} || {}\n".format(name, n_intersections, n_obstacles_per_tile))
                
            time.sleep(5)
            kill_env()
            time.sleep(10)
