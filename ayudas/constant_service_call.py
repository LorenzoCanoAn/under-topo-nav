import gazebo_msgs
import rospy
import gazebo_msgs.srv as gazebo_msgs_srv
import time


def main():
    service_name = "/gazebo/get_model_state"
    model_name = "unit_box"
    rospy.wait_for_service(service_name)
    service = rospy.ServiceProxy(service_name, gazebo_msgs_srv.GetModelState)
    request = gazebo_msgs_srv.GetModelStateRequest()
    request.model_name = model_name
    request.relative_entity_name = ""
    while not rospy.is_shutdown():
        answer = service(request)
        print(answer)
        time.sleep(0.25)


if __name__ == "__main__":
    main()
