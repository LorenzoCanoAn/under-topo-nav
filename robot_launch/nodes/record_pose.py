import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetModelStateResponse
from time import time, sleep
from threading import Thread

MODEL_NAME = "/"
FREQ = 4
POSES_FILE = "/home/lorenzo/poses.txt"


class PoseRecorderNode:
    def __init__(self):
        rospy.init_node("pose_recorder_node")
        self.proxy = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState, persistent=True)

    def request_pose(self):
        result: GetModelStateResponse = self.proxy.call(GetModelStateRequest(MODEL_NAME, ""))
        return result.pose.position

    def thread_target(self):
        with open(POSES_FILE, "w+") as f:
            while not rospy.is_shutdown():
                stime = time()
                pose = self.request_pose()
                f.write(f"{pose.x}//{pose.y}//{pose.z}\n")
                elapsed = time() - stime
                if elapsed < 1 / FREQ:
                    sleep(1 / FREQ - elapsed)

    def run(self):
        self.thread = Thread(target=self.thread_target)
        self.thread.start()
        rospy.spin()
        self.thread.join()


if __name__ == "__main__":
    node = PoseRecorderNode()
    node.run()
