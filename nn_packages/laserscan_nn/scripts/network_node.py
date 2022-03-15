#!/bin/python3
import torch
import numpy as np

import sensor_msgs.msg
import std_msgs.msg as ros_std_msg
import rospy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import scipy.ndimage

from nn_class_definitions import conv_v1, conv_v2
# insert at 1, 0 is the script path (or '' in REPL)

def from_laser_scan_to_image(laser_scan, max_range = 15):
    imgsize = 16
    laser_increment = 0.0087 #rad
    ang_start = -3.14159
    angle = ang_start - laser_increment
    image = np.zeros((imgsize*2,imgsize*2))
    for _range in (laser_scan):
        angle += laser_increment
        x = int((np.cos(angle + np.pi/2)*_range)/max_range*imgsize)
        y = int((np.sin(angle + np.pi/2)*_range)/max_range*imgsize)
        if int(np.math.sqrt(x*x + y*y))+1>= int(max_range):
            continue
        i = imgsize*2-(y+imgsize)
        j = x+imgsize
        image[i,j ]+=1
    image[imgsize,imgsize] = 0
    return (image>0)*1.0

    
def laser_tensor_to_image_tensor(laser_tensor, max_range = 15):
    if laser_tensor.shape.__len__() == 1:
        laser_tensor = torch.reshape(laser_tensor, [1,laser_tensor.shape[0]])
    laser_numpy = laser_tensor.cpu().numpy()
    images = []
    for i in range(laser_numpy.shape[0]):
        images.append(from_laser_scan_to_image(laser_numpy[i,:],max_range))
    images = np.array(images)
    images_tensor = torch.tensor(images).float().to(device)
    return images_tensor

class scan_handler:
    def __init__(self,device):
        PATH_TO_MODEL = "/home/lorenzo/catkin_ws/src/lorenzo/laser_nn/trained_nets/conv_v1__acc0.9763671875"
        self.device=device
        self.net = conv_v1()
        self.net.load_state_dict(torch.load(PATH_TO_MODEL))
        self.net.eval()
        self.net = self.net.to(device)
        self.subscriber = rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,callback=self.scan_callback)
        self.publisher = rospy.Publisher("/environment_label",ros_std_msg.String,queue_size=10)
        self.labels = ("t", "intersection", "curve", "rect", "block")

    def scan_callback(self, msg):
        ranges = msg.ranges
        ranges = np.array(ranges)
        ranges = np.where(ranges > 15, 15, ranges)  
        ranges = scipy.ndimage.median_filter(ranges,size=5)
        ranges = ranges.astype(np.double)
        ranges = torch.FloatTensor(ranges).to(self.device)
        if self.net.is_2d():
            inputs = laser_tensor_to_image_tensor(ranges)
        else:
            inputs = torch.reshape(ranges,[1,ranges.shape[0]])


        result = self.net(inputs)
        result = result.cpu().detach().numpy()
        str_msg = ros_std_msg.String()
        str_msg.data = self.labels[np.argmax(result)]
        self.publisher.publish(str_msg)
        

def main():
    

    rospy.init_node("laser_nn")
    my_scan_handler = scan_handler(device)
    rospy.loginfo("Laser network beginning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()