#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import tf

class DataSaver:
    def __init__(self):
        data = None

time_saver = DataSaver()
def send_tf_from_odometry(msg):
    br = tf.TransformBroadcaster()
    pose = msg.pose.pose
    t = rospy.Time.now()
    if time_saver.data == t:
        return
    br.sendTransform((pose.position.x,pose.position.y,pose.position.z),
                     (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w),
                     t,
                     "/base_link",
                     "/odom")
    time_saver.data = t
    

if __name__ == '__main__':
    rospy.init_node('odom_broadcaster')
    time_saver.data = rospy.Time.now()
    rospy.Subscriber("/ground_truth/state",
                     Odometry,
                     send_tf_from_odometry)
    rospy.spin()