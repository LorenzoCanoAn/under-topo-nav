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
    super_output_string = "["
    inner_string = "["
    while 1:
        msg = input()
        if msg == "b":
            break
        request = GetModelStateRequest(model_name,"")
        r = proxy.call(request)
        assert isinstance(r, GetModelStateResponse)
        p = r.pose.position
        o = r.pose.orientation
        pose_string = f"[{p.x:0.2f},{p.y:0.2f},{p.z:0.2f}],"
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