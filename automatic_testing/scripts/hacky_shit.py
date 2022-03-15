import os
import subprocess
import time

def get_pid(name):
    result = str(subprocess.run(["ps", "-e"],stdout=subprocess.PIPE).stdout).split()
    idx = result.index("{}\\n".format(name))
   

def kill_process(name):
    try:
        pid = get_pid(name)
        if pid.isnumeric():
            os.system("kill -9 {}".format(pid))
    except:
        pass

def main():
    for _ in range(20):
        print("LAUNCHING TEST")
        os.system("/bin/python3 /home/lorenzo/catkin_ws/src/lorenzo/automatic_testing/scripts/gen_w_and_launch_gazebo.py")
        print("FINISHED TEST")
        for __ in range(5):
            kill_process("python3")

if __name__ == "__main__":
    main()