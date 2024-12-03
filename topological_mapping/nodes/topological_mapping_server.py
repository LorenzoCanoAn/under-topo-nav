import rospy
from topological_mapping.msg import TopologicalMap as TopoMapMsg
from gallery_tracking.msg import TrackedGalleries
from topological_map.map import TopologicalMap


class Queue:
    def __init__(self, length):
        self.length = length
        self.data = []

    def new_data(self, newdata):
        if len(self.data) < self.length:
            self.data.append(newdata)
        else:
            self.data.pop(0)
            self.data.append(newdata)

    def get_last_interval(self, n):
        assert n < len(self.data)
        return self.data[-n:]

    def get_last(self):
        assert len(self.data) >0
        return self.data[-1]


class TopologicalMapGenerator:
    def __init__(self):
        self.map = TopologicalMap()
        self.tracked_galleries_history = Queue(20)

    def new_tracked_galleries(self, msg:TrackedGalleries):
        self.tracked_galleries_history.new_data(msg)
        self.process_tracked_galleries()

    def process_tracked_galleries(self):
        if len(self.map.nodes) ==0:
            self.init_map()
        else:
            pass


def init_map(map: TopologicalMap, msg: TrackedGalleries):
    if len(msg.angles) == 1:
        pass
    elif len(msg.angles) == 2:
        pass
    elif len(msg.angles) > 2:
        pass


class TopologicalMappingNode:
    def __init__(self):
        self.map = TopologicalMap()
        rospy.init_node("topological_mapping")
        rospy.Subscriber("/tracked_galleries", TrackedGalleries, self.tracked_galleries_callback)

    def tracked_galleries_callback(self, msg: TrackedGalleries):
        pass

    def run(self):
        pass


def main():
    node = TopologicalMappingNode()
    node.run()


if __name__ == "__main__":
    main()
