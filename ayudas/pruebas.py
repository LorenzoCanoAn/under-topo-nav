import rospy

rospy.init_node("haha")

instructions = rospy.get_param("/topological_instructions")
print(instructions)