#!/usr/bin/python3
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import rospy
from std_msgs.msg import String
from threading import Thread

class DirectionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_ros()
        self.spin_thread = Thread(target=self.ros_spin)
        self.spin_thread.start()

    def ros_spin(self):
        rospy.spin()

    def init_ros(self):
        rospy.init_node("topo_gui")
        self.init_publishers()
    
    def init_publishers(self):
        self.topo_instructions_publisher = rospy.Publisher("output_topological_instructions", String)

    def init_ui(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Create buttons
        self.left_button = QPushButton("Take Left")
        self.straight_button = QPushButton("Take Straight")
        self.right_button = QPushButton("Take Right")
        self.advance_until_node_button = QPushButton("Advance Until Node")
        self.turn_around_button = QPushButton("Turn Around")
        self.advance_10_m_button = QPushButton("Advance 10 meters")

        # Add buttons to the layout
        layout.addWidget(self.left_button)
        layout.addWidget(self.straight_button)
        layout.addWidget(self.right_button)
        layout.addWidget(self.advance_until_node_button)
        layout.addWidget(self.turn_around_button)
        layout.addWidget(self.advance_10_m_button)

        # Connect button signals to their respective slots
        self.left_button.clicked.connect(self.take_left)
        self.straight_button.clicked.connect(self.take_straight)
        self.right_button.clicked.connect(self.take_right)
        self.advance_until_node_button.clicked.connect(self.advance_until_node)
        self.turn_around_button.clicked.connect(self.turn_around)
        self.advance_10_m_button.clicked.connect(self.advance_10_m)

        # Set the layout for the widget
        self.setLayout(layout)
        self.setWindowTitle("Direction Selector")

    # Slot functions
    def take_left(self):
        self.topo_instructions_publisher.publish(String("take left"))
    def take_straight(self):
        self.topo_instructions_publisher.publish(String("take straight"))
    def take_right(self):
        self.topo_instructions_publisher.publish(String("take right"))
    def take_back(self):
        self.topo_instructions_publisher.publish(String("take back"))
    def advance_until_node(self):
        self.topo_instructions_publisher.publish(String("advance_until_node"))
    def turn_around(self):
        self.topo_instructions_publisher.publish(String("go_back"))
    def advance_10_m(self):
        self.topo_instructions_publisher.publish(String("advance_met 10"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DirectionSelector()
    window.show()
    sys.exit(app.exec_())
    window.spin_thread.join()
