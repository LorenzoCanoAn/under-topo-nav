#!/usr/bin/python3
import rospy
from gallery_tracking.msg import TrackedGalleries
import numpy as np
from std_msgs.msg import Float32, String
import actionlib
from gallery_detection_ros.msg import DetectionVectorStability
from nav_msgs.msg import Odometry
from time import time, sleep
from enum import Enum, auto


class NavigationState(Enum):
    WAITING_FOR_INSTRUCTIONS = 0
    INSTRUCTIONS_RECIEVED = 1
    SELECTING_INSTRUCTION = 2
    EXECUTING_INSTRUCTION = 3
    FINISHED_INSTRUCTION = 4
    FINISHED = 5


class GalState(Enum):
    ERROR = 0
    GALLERY = 1
    INTERSECTION = 2
    END_OF_GAL = 3


S = NavigationState


def odom_to_xyz_arr(odom: Odometry):
    p = odom.pose.pose.position
    return np.array((p.x, p.y, p.z))


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


def warp_angle_pi(angle):
    angle = angle % (np.pi * 2)
    if angle > np.pi:
        angle = np.pi * 2 - angle
    return angle


class TopologicalNavigationNode:
    def __init__(self):
        Instruction.set_node(self)
        self.active = True
        self.current_galleries: TrackedGalleries = None
        self.back_gallery_id = None
        self.followed_gallery_id = None
        self.instructions: list[Instruction] = None
        self.current_instruction_n: int = None
        self.current_instruction: Instruction = None
        self.current_state = None
        rospy.init_node("topological_navigation_node")
        self.angle_publisher = rospy.Publisher("output_angle_to_follow_topic", Float32, queue_size=1)
        self.state_publisher = rospy.Publisher("output_current_state_topic", String, queue_size=1)
        self.feedback_publisher = rospy.Publisher("output_feedback_topic", String, queue_size=1)
        self.result_publisher = rospy.Publisher("output_result_topic", String, queue_size=1)
        self.change_state(S.WAITING_FOR_INSTRUCTIONS)
        rospy.Subscriber("input_tracked_galleries_topic", TrackedGalleries, self.tracked_galleries_callback)
        rospy.Subscriber("input_stability_topic", DetectionVectorStability, self.stability_callback)
        rospy.Subscriber("input_odometry_topic", Odometry, self.odometry_callback)
        rospy.Subscriber("input_topological_instructions_topic", String, self.instructions_callback)

    def select_new_back_gallery(self):
        distances_to_back = get_distances_to_angle(np.pi, np.array(self.current_galleries.angles))
        id_back = np.argmin(distances_to_back)
        self.back_gallery_angle = self.current_galleries.angles[id_back]
        self.back_gallery_id = self.current_galleries.ids[id_back]

    def update_back_gallery(self):
        if not self.current_galleries is None:
            if len(self.current_galleries.angles) == 0:
                return
            if self.back_gallery_id in self.current_galleries.ids and np.abs(warp_angle_pi(self.back_gallery_angle)) > np.pi / 2:
                self.back_gallery_angle = self.current_galleries.angles[self.current_galleries.ids.index(self.back_gallery_id)]
            else:
                self.select_new_back_gallery()
        else:
            self.back_gallery_angle = None
            self.back_gallery_id = None

    def follow_cfg(self, force=False):
        if force:
            self.angle_publisher.publish(0)
            return True
        else:
            if self.followed_gallery_id in self.current_galleries.ids:
                angle_to_follow = self.current_galleries.angles[self.current_galleries.ids.index(self.followed_gallery_id)]
                self.angle_publisher.publish(angle_to_follow)
                return True
            return False

    def run(self):
        rospy.spin()

    def change_state(self, new_state):
        extra_str = f": {self.current_instruction}" if new_state == S.EXECUTING_INSTRUCTION else ""
        state_string = new_state.name + extra_str
        rospy.loginfo(f"Changing state to {state_string}")
        self.state_publisher.publish(state_string)
        self.state = new_state

    def parse_instructions(self, instructions: str):
        self.instructions = []
        for instruction in instructions.split(";"):
            self.instructions.append(Instruction.from_str(instruction))

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

    def set_followed_gallery_by_index(self, idx):
        zipped_gals = []
        for id_, ang in zip(self.current_galleries.ids, self.current_galleries.angles):
            zipped_gals.append((id_, ang))

        def key(element):
            return element[1] % (2 * np.pi)

        zipped_gals.sort(key=key)
        rospy.logerr("Setting followed galleries by id")
        rospy.logerr("There are currently the following galleries:")
        for id, ang in zipped_gals:
            rospy.logerr(f"id: {id}, angle: {np.rad2deg(ang)}")
        for n, (id_, ang) in enumerate(zipped_gals):
            if id_ == self.back_gallery_id:
                rospy.logerr(f"The ID of the back gallery is {id_}, with a position in the zipped array of {n}")
                n_ = (n + idx) % len(self.current_galleries.ids)
                self.set_followed_gallery(zipped_gals[n_][0])
                rospy.logerr(f"The position of the gallery will then be {n_} with an id of {zipped_gals[n_][0]}")

    def state_machine_iteration(self):
        self.update_back_gallery()
        assert not self.state == None
        if self.state == S.WAITING_FOR_INSTRUCTIONS:
            pass
        elif self.state == S.INSTRUCTIONS_RECIEVED:
            self.current_instruction_n = 0
            if len(self.current_galleries.ids) > 0:
                self.change_state(S.SELECTING_INSTRUCTION)
        elif self.state == S.SELECTING_INSTRUCTION:
            if self.current_instruction_n == len(self.instructions):
                self.current_instruction = None
                self.result_publisher.publish(String("success"))
                self.change_state(S.FINISHED)
            else:
                self.current_instruction = self.instructions[self.current_instruction_n]
                self.current_instruction.startup()
                self.feedback_publisher.publish(String(str(self.current_instruction_n)))
                self.change_state(S.EXECUTING_INSTRUCTION)
        elif self.state == S.EXECUTING_INSTRUCTION:
            result = self.current_instruction.execute()
            if result == InstructionResult.FINISHED_NOT_OK:
                self.result_publisher.publish("error")
                self.change_state(S.FINISHED)
            elif result == InstructionResult.FINISHED_OK:
                self.change_state(S.FINISHED_INSTRUCTION)
        elif self.state == S.FINISHED_INSTRUCTION:
            self.current_instruction_n += 1
            self.change_state(S.SELECTING_INSTRUCTION)
        elif self.state == S.FINISHED:
            self.change_state(S.WAITING_FOR_INSTRUCTIONS)

    def set_galstate(self):
        if len(self.current_galleries.ids) == 0:
            self.galstate = GalState.ERROR
        elif len(self.current_galleries.ids) == 1:
            self.galstate = GalState.END_OF_GAL
        elif len(self.current_galleries.ids) == 2:
            self.galstate = GalState.GALLERY
        elif len(self.current_galleries.ids) > 2:
            self.galstate = GalState.INTERSECTION

    def stability_callback(self, msg: DetectionVectorStability):
        self.is_stable = msg.is_stable

    def tracked_galleries_callback(self, msg: TrackedGalleries):
        if not self.current_galleries is None:
            self.prev_galleries = self.current_galleries
        self.current_galleries = msg
        self.set_galstate()
        self.state_machine_iteration()

    def instructions_callback(self, msg: String):
        self.parse_instructions(msg.data)
        self.change_state(S.INSTRUCTIONS_RECIEVED)
        while not self.state in [S.FINISHED, S.WAITING_FOR_INSTRUCTIONS]:
            sleep(0.1)

    def odometry_callback(self, msg: Odometry):
        if not self.current_instruction is None:
            self.current_instruction.update_odom(msg)


class OdomTracker:
    def __init__(self):
        self.prev_odom = None
        self.current_odom = None
        self.distance = 0

    def update(self, odom: Odometry):
        self.prev_odom = self.current_odom
        self.current_odom = odom_to_xyz_arr(odom)
        if not self.prev_odom is None:
            self.distance += np.linalg.norm(self.prev_odom - self.current_odom, 2)


class InstructionResult(Enum):
    FINISHED_OK = auto()
    FINISHED_NOT_OK = auto()
    NOT_FINISHED = auto()
    ERROR = auto()


class TakeState(Enum):
    STARTING = auto()
    ENTERING_GALLERY = auto()
    IN_GALLERY = auto()
    ENTERING_INTERSECTION = auto()
    IN_INTERSECTION = auto()
    EXITING_INTERSECTION_SUCCESSFULY = auto()
    EXITING_INTERSECTION_UNSUCCESSFULY = auto()
    IN_END_OF_GAL = auto()
    EXITING_END_OF_GAL = auto()


class Instruction:
    node: TopologicalNavigationNode = None

    @classmethod
    def set_node(cls, node: TopologicalNavigationNode):
        cls.node = node

    @classmethod
    def from_str(cls, string: str):
        str_split = string.split(" ")
        action = str_split[0]
        if len(str_split) > 1:
            data = str_split[1]
        else:
            data = None
        print(action)
        if not action in STR_TO_INST_CLASS.keys():
            raise AssertionError(f"{action} not in STR_TO_INST_CLASS")
        return STR_TO_INST_CLASS[action](data)

    def __init__(
        self,
        data=None,
    ):
        self.data = data

    def __str__(self):
        return self.description

    @property
    def description(self):
        return "Base instruction class"

    def execute(self) -> InstructionResult:
        result = self._execute()
        self.n_executions += 1
        return result

    def startup(self):
        self.odom_tracker = OdomTracker()
        self.start_time = time()
        self.n_executions = 0
        self.start_galstate = self.galstate

    @property
    def galstate(self):
        return self.node.galstate

    @property
    def time_elapsed(self):
        return time() - self.start_time

    @property
    def distance_done(self):
        return self.odom_tracker.distance

    def update_odom(self, odom):
        self.odom_tracker.update(odom)


class AdvanceMetInst(Instruction):
    @property
    def description(self):
        return f"Advance {self.data} meters."

    def __init__(self, data):
        data = float(data)
        super().__init__(data)

    def _execute(self):
        if self.n_executions == 0:
            self.node.set_followed_gallery_by_angle(0)
        follow_success = self.node.follow_cfg()
        if follow_success:
            if self.distance_done >= self.data:
                return InstructionResult.FINISHED_OK
            else:
                return InstructionResult.NOT_FINISHED
        else:
            return InstructionResult.ERROR


class AdvanceSecInst(Instruction):
    @property
    def description(self):
        return f"Advance {self.data} seconds."

    def __init__(self, data):
        data = float(data)
        super().__init__(data)

    def _execute(self):
        follow_success = self.node.follow_cfg()
        if follow_success:
            if self.time_elapsed >= self.data:
                return InstructionResult.FINISHED_OK
            else:
                return InstructionResult.NOT_FINISHED
        else:
            return InstructionResult.ERROR


class AdvanceUntilNodeInst(Instruction):
    counter_threshold = 5

    @property
    def description(self):
        return f"Advance until node."

    def __init__(self, data):
        super().__init__()
        self.counter = 0

    def _execute(self):
        if self.n_executions == 0:
            if self.galstate in [GalState.GALLERY, GalState.END_OF_GAL]:
                self.node.set_followed_gallery_by_angle(0)
                self.exited_starting_end_of_gal = False
                return InstructionResult.NOT_FINISHED
            else:
                return InstructionResult.ERROR
        else:
            if self.node.is_stable:
                if not self.node.follow_cfg():
                    self.node.follow_cfg(force=True)
            else:
                self.node.follow_cfg(force=True)
            if self.galstate == GalState.INTERSECTION and self.node.is_stable:
                self.counter += 1
                if self.counter > self.counter_threshold:
                    return InstructionResult.FINISHED_OK
            elif self.galstate == GalState.END_OF_GAL:
                if self.exited_starting_end_of_gal:
                    if self.node.is_stable and self.counter > self.counter_threshold:
                        return InstructionResult.FINISHED_OK
                    self.counter += 1
                else:
                    return InstructionResult.NOT_FINISHED
            elif self.galstate == GalState.GALLERY and self.node.is_stable:
                self.exited_starting_end_of_gal = True
            else:
                return InstructionResult.NOT_FINISHED


class TakeInst(Instruction):
    possible_directions = ["left", "right", "straight", "back"]
    directions_to_angle = {
        "straight": 0.0,
        "left": np.deg2rad(90.0),
        "back": np.deg2rad(180.0),
        "right": np.deg2rad(270.0),
    }
    directions_threshold = np.deg2rad(60)

    @property
    def description(self):
        return f"Take {self.data}"

    def change_state(self, new_state):
        rospy.loginfo(f"\t Changing state to: {new_state}")
        self.state = new_state

    def startup(self):
        self.change_state(TakeState.STARTING)
        return super().startup()

    def __init__(self, data):
        if data in self.possible_directions:
            pass
        else:
            data = int(data)
        super().__init__(data)

    def _execute(self):
        if self.state == TakeState.STARTING:
            if self.galstate == GalState.END_OF_GAL:
                self.node.set_followed_gallery_by_angle(0)
                self.node.follow_cfg()
            if self.galstate == GalState.GALLERY:
                self.change_state(TakeState.ENTERING_GALLERY)
            elif self.galstate == GalState.INTERSECTION:
                self.change_state(TakeState.ENTERING_INTERSECTION)
        elif self.state == TakeState.ENTERING_GALLERY:
            self.node.set_followed_gallery_by_angle(0)
            self.change_state(TakeState.IN_GALLERY)
        elif self.state == TakeState.IN_GALLERY:
            if self.node.is_stable:
                if not self.node.follow_cfg():
                    self.node.follow_cfg(force=True)
            else:
                self.node.follow_cfg(force=True)
            if self.galstate == GalState.INTERSECTION:
                self.change_state(TakeState.ENTERING_INTERSECTION)
        elif self.state == TakeState.ENTERING_INTERSECTION:
            self.node.follow_cfg(force=True)
            if self.node.is_stable and self.galstate == GalState.INTERSECTION:
                self.change_state(TakeState.IN_INTERSECTION)
        elif self.state == TakeState.IN_INTERSECTION:
            if self.try_set_gallery_from_instruction():
                self.node.follow_cfg()
                self.change_state(TakeState.EXITING_INTERSECTION_SUCCESSFULY)
            else:
                self.change_state(TakeState.EXITING_INTERSECTION_UNSUCCESSFULY)
        elif self.state == TakeState.EXITING_INTERSECTION_SUCCESSFULY:
            if not self.node.follow_cfg():
                self.node.follow_cfg(force=True)
            if self.galstate in [GalState.GALLERY, GalState.END_OF_GAL] and self.node.is_stable:
                return InstructionResult.FINISHED_OK
        elif self.state == TakeState.EXITING_INTERSECTION_UNSUCCESSFULY:
            if not self.node.follow_cfg():
                self.node.follow_cfg(force=True)
            if self.galstate == GalState.GALLERY and self.node.is_stable:
                self.change_state(TakeState.ENTERING_GALLERY)
            elif self.galstate == GalState.END_OF_GAL and self.node.is_stable:
                return InstructionResult.FINISHED_NOT_OK
        elif self.state == TakeState.IN_END_OF_GAL:
            if not self.node.follow_cfg():
                self.node.set_followed_gallery_by_angle(0)
                self.node.follow_cfg()
            if GalState == GalState.GALLERY:
                self.change_state(TakeState.ENTERING_GALLERY)
        elif self.state == TakeState.EXITING_END_OF_GAL:
            self.node.follow_cfg()
            if self.galstate == GalState.GALLERY:
                self.change_state(TakeState.ENTERING_GALLERY)
            elif self.galstate == GalState.INTERSECTION:
                self.change_state(TakeState.ENTERING_INTERSECTION)
        return InstructionResult.NOT_FINISHED

    def try_set_gallery_from_instruction(self):
        if isinstance(self.data, str):
            return self.node.set_followed_gallery_by_angle(
                self.directions_to_angle[self.data],
                threshold=self.directions_threshold,
            )
        elif isinstance(self.data, int):
            self.node.set_followed_gallery_by_index(self.data)
            return True


class StayInst(Instruction):
    @property
    def description(self):
        return f"Stay {self.data} seconds"

    def __init__(self, data):
        super().__init__(float(data))

    def _execute(self):
        if self.time_elapsed >= self.data:
            finished = True


class GoBackInst(Instruction):
    @property
    def description(self):
        return f"Go back"

    def __init__(self, *arg):
        super().__init__()

    def _execute(self):
        if self.n_executions == 0:
            self.node.set_followed_gallery_by_angle(np.pi)
        self.node.follow_cfg()
        if abs(self.node.gallery_angle_from_id(self.node.followed_gallery_id)) < np.deg2rad(10):
            return InstructionResult.FINISHED_OK
        return InstructionResult.NOT_FINISHED


STR_TO_INST_CLASS = {
    "advance_met": AdvanceMetInst,
    "advance_sec": AdvanceSecInst,
    "advance_until_node": AdvanceUntilNodeInst,
    "take": TakeInst,
    "go_back": GoBackInst,
    "stay_sec": StayInst,
}


def main():
    node = TopologicalNavigationNode()
    node.run()


if __name__ == "__main__":
    main()
