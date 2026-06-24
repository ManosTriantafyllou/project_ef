from pymycobot.myarm import MyArm
import time


def connect(port = "/dev/ttyAMA0", baudrate = 115200, timeout = 1.0):
    # create myarm object
    myarm = MyArm(port = port, baudrate = baudrate, timeout = timeout)
    time.sleep(0.5)
    # set mode: 1 always execute latest command first, 0 to execute in a queue
    myarm.set_fresh_mode(mode = 1)
    time.sleep(0.5)
    # set the arm to zero configuration
    myarm.set_encoders([2048, 2048, 2048, 2048, 2048, 2048, 2048], 50) 
    time.sleep(2.0)
    return myarm
