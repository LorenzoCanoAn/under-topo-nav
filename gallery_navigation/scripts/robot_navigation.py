#!/bin/python3
import rospy
import gazebo_msgs.srv as ros_gz_srv
from state_machine import StateMachine, State
from helper_classes import MoveBaseHandler, NNOutputHandler


class TunnelNavigationSM(StateMachine):
    states_ = []

    def __init__(self, instruction_list):
        self.instruction_list = instruction_list
        self.init_transition_variables()
        self.n_instruction = 0
        self.current_state_name = "start_state"
        self.robot_handler = MoveBaseHandler()
        self.nn_handler = NNOutputHandler()
        self.start_signal = self.nn_handler.has_first_callback_happened
        for s in self.states_:
            self.states[s.name] = s(self)
        super(TunnelNavigationSM, self).__init__()

    def run_as_independent_loop(self):
        while not self.start_signal():
            pass
        while not self.current_state.is_final and not rospy.is_shutdown():
            self.do_one_step()
        self.current_state.state_action()


    def init_transition_variables(self):
        self.tv_instruction_number = 0
        self.tv_final_pose_x = rospy.get_param(
            "/final_blocker_position_x", default=37.5)
        self.tv_final_pose_y = rospy.get_param(
            "/final_blocker_position_y", default=30.5)
        self.tv_has_left_intersection = False

    def update_transition_variables(self):
        self.tv_current_situation = self.nn_handler.situation
        self.tv_valid_directions = self.nn_handler.valid_directions
        self.tv_move_base_active = self.robot_handler.move_base_active
        self.tv_quadrants = self.nn_handler.quadrants



class TunnelNavigationState(State):

    def __init__(self, sm: TunnelNavigationSM):
        self.sm = sm
        self.next_state_name = self.name
        self.set_up_decission_variables()

    def go_to_goal_from_angle(self, angle, distance=1):
        self.sm.robot_handler.send_goal_from_angle(angle, distance=distance)

    def set_up_decission_variables(self):
        self.decission_variables = []
    def set_transition(self, next_state_name):
        super().set_transition(next_state_name)
        rospy.loginfo("Next State: {}".format(next_state_name))


class StartState(TunnelNavigationState):
    name = "start_state"

    def state_action(self):
        self.set_transition(
            "determining_situation")


class DetermineSituationState(TunnelNavigationState):
    name = "determining_situation"

    def state_action(self):
        self.set_transition(self.sm.tv_current_situation)


class InRectState(TunnelNavigationState):
    name = "in_rect"
    direction_priorities = ["front", "right", "left"]

    def state_action(self):
        error = True
        for preffered_direction in self.direction_priorities:
            if preffered_direction in self.sm.tv_valid_directions:
                error = False
                angle_of_goal = self.sm.tv_quadrants[preffered_direction]
                self.go_to_goal_from_angle(angle_of_goal)
                self.set_transition("advancing_in_rect")
                break
        if error:
            self.set_transition("failed")


class AdvancingInRectState(TunnelNavigationState):
    name = "advancing_in_rect"

    def state_action(self):
        if self.move_base_started:
            if not self.sm.tv_move_base_active:
                self.set_transition("determining_situation")
        else:
            if self.sm.tv_move_base_active:
                self.move_base_started = True

    def when_transitioned_do(self):
        self.move_base_started = False


class EndOfGalleryState(TunnelNavigationState):
    name = "in_end_of_gallery"
    direction_priorities = ["front", "right", "left", "back"]
    get_robot_pose_client = rospy.ServiceProxy(
        "/gazebo/get_model_state", ros_gz_srv.GetModelState)

    def state_action(self):
        if self.sm.tv_instruction_number == 0:

            self.set_transition("going_back")
        else:
            if self.get_distance_from_final_position() > 10:
                self.set_transition("failed")
            else:
                self.set_transition("success")

    def get_distance_from_final_position(self):
        robot_pose_msg = self.get_robot_pose_client.call(
            ros_gz_srv.GetModelStateRequest("/", ""))
        rx = robot_pose_msg.pose.position.x
        ry = robot_pose_msg.pose.position.y
        return ((self.sm.tv_final_pose_x-rx)**2+(self.sm.tv_final_pose_y-ry)**2)**0.5


class GoingBackState(TunnelNavigationState):
    name = "going_back"

    def state_action(self):
        if self.move_base_started:
            if not self.sm.tv_move_base_active:
                self.set_transition("determining_situation")
        else:
            if self.sm.tv_move_base_active:
                self.move_base_started = True

    def when_transitioned_do(self):
        self.move_base_started = False


class InNodeState(TunnelNavigationState):
    name = "in_node"

    def state_action(self):
        if self.sm.tv_instruction_number == self.sm.instruction_list.__len__():
            self.set_transition("failed")
        else:
            instruction = self.sm.instruction_list[
                self.sm.tv_instruction_number]
            if instruction in self.sm.tv_valid_directions:
                self.sm.tv_instruction_number += 1
                self.go_to_goal_from_angle(
                    self.sm.tv_quadrants[instruction])
                self.set_transition("going_to_exit")
            else:
                self.set_transition("failed")


class GoingToExitState(TunnelNavigationState):
    name = "going_to_exit"

    def state_action(self):
        if self.move_base_started:
            if not self.sm.tv_move_base_active:
                self.set_transition("checking_exit")
        else:
            if self.sm.tv_move_base_active:
                self.move_base_started = True

    def when_transitioned_do(self):
        self.move_base_started = False


class CheckingExitState(TunnelNavigationState):
    name = "checking_exit"

    def state_action(self):
        if self.sm.tv_current_situation == "in_node":
            self.go_to_goal_from_angle(
                self.sm.tv_quadrants["front"])
            self.set_transition("going_to_exit")
        else:
            self.set_transition("determining_situation")


class SuccessState(TunnelNavigationState):
    name = "success"
    is_final = True

    def state_action(self):
        rospy.set_param("/test_status","success")
        rospy.signal_shutdown("The robot has reached it's destination")


class FailedState(TunnelNavigationState):
    name = "failed"
    is_final = True

    def state_action(self):
        rospy.set_param("/test_status","fail")
        rospy.signal_shutdown("The robot has failed to reach it's destination")


TunnelNavigationSM.states_ = [StartState, DetermineSituationState, InRectState, AdvancingInRectState,
                              EndOfGalleryState, GoingBackState, SuccessState, FailedState, InNodeState, GoingToExitState, CheckingExitState]


def main():
    rospy.init_node("robot_control_node")
    #instructions = rospy.get_param("/instructions")
    instructions=["front","right"]
    print(instructions)
    sm = TunnelNavigationSM(instructions)
    sm.run_as_independent_loop()
    rospy.spin()


if __name__ == "__main__":
    main()
