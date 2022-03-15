import subprocess

def kill_process_and_subprocess(pid):
    result = subprocess.run(["pstree", "-p", f"{pid}"],capture_output=True)
    result = str(result.stdout)
    result = result.replace(" ","")
    list_of_numbers = set()
    for n, i in enumerate(result):
        if i == "(":
            start = n +1
        if i == ")":
            list_of_numbers.add(int(result[start:n]))
    for pid in list_of_numbers:
        result = subprocess.run(["kill", "-9", str(pid)], capture_output=True)
import os


def kill_process(name):
    try:
        pid = get_pid(name)
        if pid.isnumeric():
            os.system("kill -9 {}".format(pid))
    except:
        pass

def get_pid(name):
    result = str(subprocess.run(["ps", "-e"],stdout=subprocess.PIPE).stdout).split()
    idx = result.index("{}\\n".format(name))

    return result[idx-3]
    
def kill_master():
    kill_process("rosmaster")

def kill_ros_nodes():
    os.system("rosnode kill -all")

def kill_gazebo():
    for _ in range(1):
        kill_process("gzserver")
    for _ in range(1):
        kill_process("gzclient")

def kill_env():
    for _ in range(4):
        kill_process("roslaunch")
        kill_process("ekf_localizatio")
        kill_process("marker_server")
        kill_process("twist_mux")
        kill_process("joy_node")
        kill_process("teleop_node")
        kill_process("pointcloud_to_l")
        kill_process("pointcloud_to_i")
        kill_process("network_node.py")
        kill_process("move_base")
        kill_process("robot_state_pub")
        kill_process("roscore")
        kill_process("rosmaster")
        kill_process("rosout")
        kill_process("goal_generation")
        kill_process("roslaunch <defunct>")
        kill_process("simple_heading_")
        kill_process("simple_heading")

        kill_ros_nodes()
        kill_master()
        kill_gazebo()

        kill_process("python3")
