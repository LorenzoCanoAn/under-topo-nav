#!/usr/bin/env python3

# stdlib
import argparse
import os
from copy import deepcopy

# 3rd-party
import rosbag
from tqdm import tqdm
from colorama import Fore, Style
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Transform
from pytictoc import TicToc

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Parse command line arguments
    # --------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-bfi", "--bagfile_in", help="Full path to the input bagfile", type=str, required=False
    )
    ap.add_argument(
        "-bfo",
        "--bagfile_out",
        help="Full path to output the bagfile. If not given will be named aggregated.bag and placed on the same folder as the input bag.",
        type=str,
    )
    ap.add_argument("-f", "--folder", help="Folder containing rosbag files", type=str, default=None)

    args = vars(ap.parse_args())

    if args["bagfile_out"] is None and not args["bagfile_in"] is None:
        path = os.path.dirname(args["bagfile_in"])
        print(path)
        args["bagfile_out"] = path + "/aggregated.bag"

    if args["folder"] is None:
        bagfiles_in = [args["bagfile_in"]]
        bagfiles_out = [args["bagfile_out"]]
    else:
        files_in_dir = os.listdir(args["folder"])
        bagfiles_in = []
        bagfiles_out = []
        for filename in files_in_dir:
            if filename[-4:] == ".bag":
                bagfiles_in.append(os.path.join(args["folder"], filename))
                os.makedirs(os.path.join(args["folder"], "fixed"), exist_ok=True)
                bagfiles_out.append(os.path.join(args["folder"], "fixed", filename))

    for bagfile_in, bagfile_out in zip(bagfiles_in, bagfiles_out):
        print(bagfile_in)
        # --------------------------------------------------------------------------
        # Initial setup
        # --------------------------------------------------------------------------
        tictoc = TicToc()
        tictoc.tic()
        bag_out = rosbag.Bag(bagfile_out, "w")

        # --------------------------------------------------------------------------
        # Read the bag input file
        # --------------------------------------------------------------------------
        bag_file = bagfile_in
        print("Loading bagfile " + bag_file)
        bag = rosbag.Bag(bag_file)  # load the bag file
        bag_info = bag.get_type_and_topic_info()
        bag_types = bag_info[0]
        bag_topics = bag_info[1]

        # --------------------------------------------------------------------------
        # Get initial stamp to compute mission time
        # --------------------------------------------------------------------------
        for topic, msg, stamp in bag.read_messages():
            initial_stamp = stamp
            break

        # --------------------------------------------------------------------------
        # Get all transforms with tf_static
        # --------------------------------------------------------------------------
        def generateTransformKey(transform):
            return transform.header.frame_id + "_to_" + transform.child_frame_id

        transforms_dict = (
            {}
        )  # a dicts of all transforms in the tf_static topics. Keys are strings with <parent>_to_<child> patterns

        print("Searching for msgs on topic /tf_static. Please wait...")
        for topic, msg, stamp, connection_header in tqdm(
            bag.read_messages(return_connection_header=True),
            total=bag.get_message_count(),
            desc="Processing bag messages",
        ):
            if topic == "/tf_static":
                for transform in msg.transforms:
                    key = generateTransformKey(transform)
                    transforms_dict[key] = transform

        print("Found " + str(len(transforms_dict.keys())) + " static transforms in the bagfile.")

        # Create an aggregate /tf_static message to publish instead of all others
        static_transform_msg = TFMessage()
        for key, transform in transforms_dict.items():
            static_transform_msg.transforms.append(transform)

        # Get a connection_header for a tf_static msg
        there_is_static_tf = False
        for topic, msg, stamp, connection_header in bag.read_messages(
            return_connection_header=True
        ):
            if topic == "/tf_static":
                there_is_static_tf = True
                tf_static_connection_header = connection_header
                break
        if not there_is_static_tf:
            os.remove(bagfile_out)
            continue

        # --------------------------------------------------------------------------
        # Writing new bagfile
        # --------------------------------------------------------------------------
        print("Producing bagfile. Please wait...")
        bag_out.write(
            "/tf_static",
            static_transform_msg,
            initial_stamp,
            connection_header=tf_static_connection_header,
        )
        current_stamp = initial_stamp
        for topic, msg, stamp, connection_header in tqdm(
            bag.read_messages(return_connection_header=True),
            total=bag.get_message_count(),
            desc="Processing bag messages",
        ):
            mission_time = stamp - initial_stamp

            if not topic == "/tf_static":
                bag_out.write(topic, msg, stamp, connection_header=connection_header)

            if topic == "/tf":
                if stamp > current_stamp:
                    bag_out.write(
                        "/tf_static",
                        static_transform_msg,
                        stamp,
                        connection_header=tf_static_connection_header,
                    )
                    current_stamp = stamp

        bag.close()  # close the bag file.
        bag_out.close()  # close the bag file.

        # Print final report
        print("Finished in " + str(round(tictoc.tocvalue(), 2)) + " seconds.")
        print(
            "Created new bagfile "
            + Fore.BLUE
            + bagfile_out
            + Style.RESET_ALL
            + " in "
            + str(round(tictoc.tocvalue(), 2))
            + " seconds."
        )
