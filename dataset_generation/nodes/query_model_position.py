#!/usr/bin/python3
from black import out
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse, GetModelStateRequest
import time


def main():
    model_name = "Untitled"
    rospy.init_node(f"model_query_{model_name}")
    
    # Setup service
    rospy.wait_for_service("/gazebo/get_model_state")
    proxy = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    msg = ""
    output_string = "["
    while 1:
        msg = input()
        if msg != "":
            break
        request = GetModelStateRequest(model_name,"")
        r = proxy.call(request)
        assert isinstance(r, GetModelStateResponse)
        p = r.pose.position
        o = r.pose.orientation
        pose_string = f"[{p.x},{p.y},{p.z}],"
        print(pose_string)
        output_string += pose_string
    output_string += "]"
    print(output_string)



if __name__ == "__main__":
    main()