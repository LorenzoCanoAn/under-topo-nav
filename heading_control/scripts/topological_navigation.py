import rospy
from gallery_tracking.msg import TrackedGalleries
import numpy as np
from std_msgs.msg import Float32, Bool, String
from heading_control.msg import topological_instructions
from gallery_detection_ros.msg import DetectionVectorStability
from nav_msgs.msg import Odometry
from time import time

WAITING_FOR_INSTRUCTIONS = 0
INSTRUCTIONS_RECIEVED = 1
SELECTING_INSTRUCTION = 2
EXECUTING_INSTRUCTION = 3
FINISHED_INSTRUCTION = 4
FINISHED = 5

STATE_NAME = {
    0: "WAITING_FOR_INSTRUCTIONS",
    1: "INSTRUCTIONS_RECIEVED",
    2: "SELECTING_INSTRUCTION",
    3: "EXECUTING_INSTRUCTION",
    4: "FINISHED_INSTRUCTION",
    5: "FINISHED",
}


def is_int(val, base=10):
    try:
        result = int(val, base)
    except ValueError:
        result = False
    return bool(result)


def is_float(val):
    try:
        result = float(val)
    except ValueError:
        result = False
    return bool(result) and not is_int(val)


def get_distances_to_angle(angle: float, angles: np.ndarray):
    distances = (angles - angle) % (2 * np.pi)
    indexer = np.where(distances > np.pi)
    distances[indexer] = 2 * np.pi - distances[indexer]
    return distances


def warp_angle(angle):
    angle = angle % (np.pi * 2)
    if angle > np.pi:
        angle = np.pi * 2 - angle
    return angle


class TopologicalNavigationNode:
    def __init__(self):
        self.active = True
        self.current_galleries: TrackedGalleries = None
        self.prev_galleries: TrackedGalleries = None
        self.current_odom: Odometry = None
        self.prev_odom: Odometry = None
        self.back_gallery_id = None
        self.followed_gallery_id = None
        self.instructions: list[Instruction] = None
        self.current_instruction_n: int = None
        self.current_instruction: Instruction = None
        rospy.init_node("topological_navigation_node")
        self.angle_publisher = rospy.Publisher("angle_to_follow", Float32, queue_size=1)
        self.state_publisher = rospy.Publisher("/current_state", String, queue_size=1)
        self.change_state(WAITING_FOR_INSTRUCTIONS)
        rospy.Subscriber("tracked_galleries", TrackedGalleries, self.tracked_galleries_callback)
        rospy.Subscriber(
            "topological_instructions",
            topological_instructions,
            self.topological_instructions_callback,
        )
        rospy.Subscriber("/is_detection_stable", DetectionVectorStability, self.stability_callback)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odometry_callback)

    def tracked_galleries_callback(self, msg: TrackedGalleries):
        if not self.current_galleries is None:
            self.prev_galleries = self.current_galleries
        self.current_galleries = msg
        self.state_machine_iteration()

    def topological_instructions_callback(self, msg: topological_instructions):
        self.parse_instructions(msg)
        self.change_state(INSTRUCTIONS_RECIEVED)

    def odometry_callback(self, msg: Odometry):
        if not self.current_odom is None:
            self.prev_odom = self.current_odom
        self.current_odom = msg
        if not self.current_instruction is None:
            self.current_instruction.update_odom()

    def select_back_gallery(self):
        distances_to_back = get_distances_to_angle(np.pi, np.array(self.current_galleries.angles))
        id_back = np.argmin(distances_to_back)
        self.back_gallery_angle = self.current_galleries.angles[id_back]
        self.back_gallery_id = self.current_galleries.ids[id_back]

    def update_back_gallery(self):
        if not self.current_galleries is None:
            if self.back_gallery_id in self.current_galleries.ids:
                self.back_gallery_angle = self.current_galleries.angles[
                    self.current_galleries.ids.index(self.back_gallery_id)
                ]
            else:
                self.select_back_gallery()
        else:
            self.back_gallery_angle = None
            self.back_gallery_id = None

    def follow_gallery(self, force=False):
        if force:
            self.angle_publisher.publish(0)
            return True
        else:
            if self.followed_gallery_id in self.current_galleries.ids:
                angle_to_follow = self.current_galleries.angles[
                    self.current_galleries.ids.index(self.followed_gallery_id)
                ]
                self.angle_publisher.publish(angle_to_follow)
                return True
            return False

    def run(self):
        rospy.spin()

    def change_state(self, new_state):
        extra_str = f": {self.current_instruction}" if new_state == EXECUTING_INSTRUCTION else ""
        state_string = STATE_NAME[new_state] + extra_str
        rospy.loginfo(f"Changing state to {state_string}")
        self.state_publisher.publish(state_string)
        self.state = new_state

    def parse_instructions(self, instructions: topological_instructions):
        self.instructions = []
        for instruction in instructions.instructions:
            self.instructions.append(Instruction.from_str(self, instruction))

    def gallery_id_closest_to_angle(self, angle, threshold=False):
        distances = get_distances_to_angle(angle, np.array(self.current_galleries.angles))
        idx_min = np.argmin(distances)
        if not threshold is None:
            if distances[idx_min] > threshold:
                return None
        return self.current_galleries.ids[idx_min]

    def gallery_angle_from_id(self, id):
        return self.current_galleries.angles[self.current_galleries.ids.index(id)]

    def set_followed_gallery(self, gal_id):
        self.followed_gallery_id = gal_id

    def set_followed_gallery_by_angle(self, angle, threshold=None):
        _id = self.gallery_id_closest_to_angle(angle, threshold)
        if _id is None:
            return False
        else:
            self.set_followed_gallery(_id)
            return True

    def state_machine_iteration(self):
        self.update_back_gallery()
        assert not self.state == None
        if self.state == WAITING_FOR_INSTRUCTIONS:
            pass
        elif self.state == INSTRUCTIONS_RECIEVED:
            self.current_instruction_n = 0
            self.change_state(SELECTING_INSTRUCTION)
        elif self.state == SELECTING_INSTRUCTION:
            if self.current_instruction_n == len(self.instructions):
                self.current_instruction = None
                self.change_state(FINISHED)
            else:
                self.current_instruction = self.instructions[self.current_instruction_n]
                self.current_instruction.startup()
                self.change_state(EXECUTING_INSTRUCTION)
        elif self.state == EXECUTING_INSTRUCTION:
            finished = self.current_instruction.execute()
            if finished:
                self.change_state(FINISHED_INSTRUCTION)
        elif self.state == FINISHED_INSTRUCTION:
            self.current_instruction_n += 1
            self.change_state(SELECTING_INSTRUCTION)
        elif self.state == FINISHED:
            self.change_state(WAITING_FOR_INSTRUCTIONS)

    def stability_callback(self, msg: DetectionVectorStability):
        self.is_stable = msg.is_stable


class Instruction:
    posible_actions = [
        "advance_met",
        "advance_sec",
        "advance_min",
        "take",
        "stay_sec",
    ]
    possible_directions = ["left", "right", "straight", "back"]
    directions_to_angle = {
        "straight": 0.0,
        "left": np.deg2rad(90.0),
        "back": np.deg2rad(180.0),
        "right": np.deg2rad(270.0),
    }
    directions_threshold = np.deg2rad(60)

    def __init__(
        self,
        node: TopologicalNavigationNode,
        action,
        data=None,
    ):
        self.node = node
        self.action = action
        self.data = data

    def __str__(self):
        return self.action + " " + str(self.data)

    @classmethod
    def from_str(self, node: TopologicalNavigationNode, string: str):
        str_split = string.split(" ")
        action = str_split[0]
        data = str_split[1]
        try:
            assert action in self.posible_actions
        except:
            assert action in self.posible_actions
        if action == "advance_met":
            assert is_float(data)
            data = float(data)
        elif action == "advance_sec":
            assert is_float(data)
            data = float(data)
        elif action == "advance_min":
            assert is_float(data)
            action = "advance_sec"
            data = float(data) * 60
        elif action == "take":
            if is_int(data):
                data = int(action)
            else:
                try:
                    assert data in self.possible_directions
                except:
                    rospy.logerr(f"{data} not a valid direction")
        elif action == "stay_sec":
            assert data.isnumeric()
            data = float(data)

        return Instruction(
            node,
            action,
            data,
        )

    def startup(self):
        if self.action in ["advance_sec", "advance_met"]:
            self.node.set_followed_gallery_by_angle(0)
        self.distance = 0
        self.n_executions = 0
        self.start_time = time()
        self.initial_galleries = self.node.current_galleries
        if len(self.initial_galleries.ids) > 2:
            self.change_state("entering_intersection")
        elif len(self.initial_galleries.ids) == 2:
            self.change_state("entering_gallery")
        elif len(self.initial_galleries.ids) == 1:
            self.change_state("in_end_of_gallery")

    @property
    def time_elapsed(self):
        return time() - self.start_time

    def change_state(self, new_state):
        rospy.loginfo(f"\t Changing state to: {new_state}")
        self.state = new_state

    def execute(self):
        finished = False
        if self.action == "advance_sec":
            success = self.node.follow_gallery()
            if success:
                if self.time_elapsed >= self.data:
                    finished = True
            else:
                finished = True
        if self.action == "advance_met":
            success = self.node.follow_gallery()
            if success:
                if self.distance >= self.data:
                    finished = True
            else:
                finished = True
        if self.action == "take":
            if len(self.node.current_galleries.ids) == 1:
                current_state = "end_of_gallery"
            elif len(self.node.current_galleries.ids) == 2:
                current_state = "gallery"
            elif len(self.node.current_galleries.ids) > 2:
                current_state = "intersection"
            if isinstance(self.data, str):
                if self.data == "back":
                    if not hasattr(self, "new_gallery_id"):
                        result = self.node.set_followed_gallery_by_angle(
                            self.directions_to_angle[self.data]
                        )
                        self.new_gallery_id = self.node.followed_gallery_id
                    self.node.follow_gallery()
                    if self.node.gallery_angle_from_id(self.new_gallery_id) % (
                        np.pi * 2
                    ) < np.deg2rad(10):
                        finished = True
                else:
                    if self.state == "in_end_of_gallery":
                        # In this state turn around
                        self.node.set_followed_gallery_by_angle(0)
                    elif self.state == "exiting_end_of_gallery":
                        self.node.follow_gallery()
                        if current_state == "gallery":
                            self.change_state("entering_gallery")
                    elif self.state == "entering_gallery":
                        self.node.set_followed_gallery_by_angle(0)
                        self.change_state("in_gallery")
                    elif self.state == "in_gallery":
                        if self.node.is_stable:
                            if not self.node.follow_gallery():
                                self.node.follow_gallery(force=True)
                        else:
                            self.node.follow_gallery(force=True)
                        if current_state == "intersection":
                            self.change_state("entering_intersection")
                    elif self.state == "entering_intersection":
                        self.node.follow_gallery(force=True)
                        if self.node.is_stable:
                            self.change_state("in_intersection")
                    elif self.state == "in_intersection":
                        if self.try_set_gallery_from_instruction():
                            self.node.follow_gallery()
                            self.change_state("exiting_intersection_successfully")
                        else:
                            self.change_state("exiting_intersection_unsuccessfully")
                    elif self.state == "exiting_intersection_successfully":
                        if not self.node.follow_gallery():
                            self.node.follow_gallery(force=True)
                        if current_state == "gallery" and self.node.is_stable:
                            self.change_state("finished")
                            finished = True
                    elif self.state == "exiting_intersection_unsuccessfully":
                        self.node.follow_gallery()
                        if current_state == "gallery" and self.node.is_stable:
                            self.change_state("entering_gallery")

            elif isinstance(self.data, int):
                pass
        if self.action == "stay_sec":
            if self.time_elapsed >= self.data:
                finished = True
        self.n_executions += 1
        return finished

    def try_set_gallery_from_instruction(self):
        return self.node.set_followed_gallery_by_angle(
            self.directions_to_angle[self.data],
            threshold=self.directions_threshold,
        )

    def update_odom(self):
        p1 = self.node.current_odom.pose.pose.position
        p2 = self.node.prev_odom.pose.pose.position
        p1 = np.array([p1.x, p1.y, p1.z])
        p2 = np.array([p2.x, p2.y, p2.z])
        self.distance += np.linalg.norm(p1 - p2, ord=2)


def main():
    node = TopologicalNavigationNode()
    node.run()


if __name__ == "__main__":
    main()
