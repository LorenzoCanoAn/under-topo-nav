#!/bin/python3

from std_msgs.msg import Float32, Float32MultiArray, Bool, String
from gallery_tracking.msg import TrackedGalleries
from gallery_detection_ros.msg import DetectionVector, DetectionVectorStability, DetectedGalleries
import rospy
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import math
import threading

PI = math.pi
import io
import cv_bridge
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image


class Plotter:
    def __init__(self):
        self._updated_detected_galleries = False
        self._updated_nn_output = False
        self._updated_filtered_detection_vector = False
        self._updated_tracked_galleries = False
        self._updated_back_gallery = False
        self._updated_followed_gallery = False
        self._updated_corrected_angle = False
        self._updated_stability = False
        self._plot_on_window = rospy.get_param("~plot_on_window", False)
        self._plot_on_rviz = rospy.get_param("~plot_on_rviz", True)
        self._polar = rospy.get_param("~polar", True)
        rospy.Subscriber(
            "/gallery_detection_vector",
            DetectionVector,
            callback=self.gallery_detection_vector_callback,
        )
        rospy.Subscriber(
            "/filtered_detection_vector",
            DetectionVector,
            callback=self.filtered_detection_vector_callback,
        )
        rospy.Subscriber(
            "/currently_detected_galleries",
            DetectedGalleries,
            callback=self.detected_galleries_callback,
        )
        rospy.Subscriber(
            "/tracked_galleries",
            TrackedGalleries,
            callback=self.tracked_galleries_callback,
        )
        rospy.Subscriber("/back_gallery", Float32, callback=self.back_gallery_callback)
        rospy.Subscriber("/angle_to_follow", Float32, callback=self.followed_gallery_callback)
        rospy.Subscriber("/corrected_angle", Float32, callback=self.corrected_angle_callback)
        rospy.Subscriber("/is_detection_stable", DetectionVectorStability, callback=self.stability_callback)
        rospy.Subscriber("/current_state", String, callback=self.current_state)
        if self._plot_on_rviz:
            self.publisher = rospy.Publisher("/nn_plot", ImageMsg, queue_size=1)
        self.draw_loop()

    def gallery_detection_vector_callback(self, msg: DetectionVector):
        gallery_detection_vector = np.array(msg.vector)
        angles = np.linspace(0, 2 * PI, len(gallery_detection_vector))
        self._gallery_detection_vector = gallery_detection_vector
        self._angles = angles
        self._updated_nn_output = True

    def filtered_detection_vector_callback(self, msg: DetectionVector):
        filtered_detection_vector = np.array(msg.vector)
        self._filtered_detection_vector = filtered_detection_vector
        self._updated_filtered_detection_vector = True

    def detected_galleries_callback(self, msg: DetectedGalleries):
        angles = np.array(msg.angles).reshape((-1, 1))
        values = np.array(msg.values).reshape((-1, 1))
        assert len(angles) == len(values)
        self._detected_galleries = np.hstack((angles, values))
        self._updated_detected_galleries = True

    def tracked_galleries_callback(self, msg: TrackedGalleries):
        angles = np.array(msg.angles).reshape((-1, 1))
        values = np.ones_like(angles)
        self._tracked_galleries_angles = np.hstack((angles, values))
        self._tracked_galleries_ids = msg.ids
        self._updated_tracked_galleries = True

    def back_gallery_callback(self, msg: Float32):
        self._back_gallery = np.array([msg.data, 0.9])
        self._updated_back_gallery = True

    def followed_gallery_callback(self, msg: Float32):
        self._followed_gallery = np.array([msg.data, 0.9])
        self._updated_followed_gallery = True

    def corrected_angle_callback(self, msg: Float32):
        self._corrected_angle = np.array([msg.data, 0.8])
        self._updated_corrected_angle = True

    def stability_callback(self, msg: DetectionVectorStability):
        self._is_stable = msg.is_stable
        self._updated_stability = True

    def current_state(self, msg: String):
        print(msg)
        self._ax1.set_title(msg.data, pad=13)

    def draw_loop(self):
        # INIT THE PYPLOT VARIABLES
        if self._polar:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig = plt.figure(figsize=(10, 5))
        self._ax1 = fig.add_subplot(111, polar=self._polar)
        (self._nn_output_lines,) = self._ax1.plot([], lw=6, c="b")
        (self._nn_filtered_vector_lines,) = self._ax1.plot([], lw=6, c="k")
        self._currently_detected_scatter = self._ax1.scatter([], [], c="b", s=300)
        self._tracked_galleries_scatter = self._ax1.scatter([], [], c="r", s=300)
        self._back_gallery_scatter = self._ax1.scatter([], [], color="k", s=500, marker="P")
        self._followed_gallery_scatter = self._ax1.scatter([], [], color="c", s=500, marker="P")
        self._corrected_angle_scatter = self._ax1.scatter([], [], color="g", s=500, marker="P")
        self._tracked_galleries_id_text: list[Text] = [self._ax1.text(0, 0, "", fontsize="xx-large") for _ in range(10)]
        self._ax1.tick_params(labelsize=20)
        self._ax1.set_ylim([0, 1.3])
        if self._polar:
            self._ax1.set_theta_zero_location("N")
        else:
            self._ax1.set_xlim([0, 2 * np.pi])
        fig.canvas.draw()
        self._ax1background = fig.canvas.copy_from_bbox(self._ax1.bbox)
        if self._plot_on_window:
            plt.show(block=False)
        if self._plot_on_rviz:
            self.bridge = cv_bridge.CvBridge()

        while not rospy.is_shutdown():
            # set new drawings
            # NN OUTPUT
            if self._updated_nn_output:
                self._nn_output_lines.set_data(self._angles, self._gallery_detection_vector)
            if self._updated_filtered_detection_vector:
                self._nn_filtered_vector_lines.set_data(self._angles, self._filtered_detection_vector)
            if self._updated_detected_galleries:
                self._currently_detected_scatter.set_offsets(self._detected_galleries)
            if self._updated_tracked_galleries:
                self._tracked_galleries_scatter.set_offsets(self._tracked_galleries_angles)
                for text_obj in self._tracked_galleries_id_text:
                    text_obj.set_visible(False)
                for n in range(len(self._tracked_galleries_ids)):
                    tg_angle, tg_value = self._tracked_galleries_angles[n]
                    tg_id = self._tracked_galleries_ids[n]
                    text_obj = self._tracked_galleries_id_text[n]
                    text_obj.set_visible(True)
                    text_obj.set_text(str(tg_id))
                    text_obj.set_position((tg_angle, 1 + 0.2))

            if self._updated_back_gallery:
                self._back_gallery_scatter.set_offsets(self._back_gallery)
            if self._updated_followed_gallery:
                self._followed_gallery_scatter.set_offsets(self._followed_gallery)
            if self._updated_corrected_angle:
                self._corrected_angle_scatter.set_offsets(self._corrected_angle)
            # fig.canvas.restore_region(self._ax1background)
            if self._updated_stability:
                if self._is_stable:
                    self._ax1.set_facecolor("lightgreen")
                else:
                    self._ax1.set_facecolor("lightcoral")

            # restore background
            # redraw the datapoints
            self._ax1.draw_artist(self._nn_output_lines)
            self._ax1.draw_artist(self._nn_filtered_vector_lines)
            self._ax1.draw_artist(self._currently_detected_scatter)
            self._ax1.draw_artist(self._tracked_galleries_scatter)
            self._ax1.draw_artist(self._back_gallery_scatter)
            self._ax1.draw_artist(self._followed_gallery_scatter)
            self._ax1.draw_artist(self._corrected_angle_scatter)
            for tx_obj in self._tracked_galleries_id_text:
                self._ax1.draw_artist(tx_obj)
            # fill the axes rectangle
            if self._plot_on_window:
                fig.canvas.blit(self._ax1.bbox)
                fig.canvas.flush_events()
            if self._plot_on_rviz:
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                pil_img = Image.open(buf)
                pil_img.convert("RGB")
                opencv_img = np.array(pil_img)[:, :, :3][:, :, ::-1]
                self.publisher.publish(self.bridge.cv2_to_imgmsg(opencv_img))


def main():
    rospy.init_node("plotter")
    rospy_thread = threading.Thread(target=rospy.spin).start()
    plotter = Plotter()
    rospy_thread.join()


if __name__ == "__main__":
    main()
