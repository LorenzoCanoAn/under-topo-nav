#!/usr/bin/python3
from black import out
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse, GetModelStateRequest
import time


def main():
    ref_name = "Untitled"
    measure_name = "Untitled_clone"
    rospy.init_node(f"distance_measureing")
    
    # Setup service
    rospy.wait_for_service("/gazebo/get_model_state")
    proxy = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    msg = ""
    super_output_string = "["
    inner_string = "["
    while 1:
        
        r_ref = GetModelStateRequest(ref_name,"")
        meas_req = GetModelStateRequest(measure_name,"")
        r_ref = proxy.call(r_ref)
        r_meas = proxy.call(meas_req)

        p_ref = r_ref.pose.position
        p_meas = r_meas.pose.position
        pose_string = f"[{p_ref.x - p_meas.x:0.2f},{p_ref.y- p_meas.y:0.2f},{p_ref.z- p_meas.z:0.2f}],"
        print(pose_string)
        inner_string += pose_string
        if msg == "a":
            inner_string += "],"
            super_output_string += inner_string
            print(inner_string)
            inner_string = "["

    super_output_string += "]"
    print(super_output_string)



if __name__ == "__main__":
    main()