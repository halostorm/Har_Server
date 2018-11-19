import os
import time
import struct
from socket import *
def run():
    print("run")
    return True 

def send_event(event_id,event_value,fps):
    try:
        cliet_socket = socket(AF_INET, SOCK_STREAM)
        cliet_socket.connect(('127.0.0.1',10004))
        data=struct.pack('2i',event_id,event_value)
        cliet_socket.send(data)
        time.sleep(1);
        cliet_socket.close()
        print("send_event ok")
        return True
    except:
        print("send_event failed!")
        return True

def kill():
    print("kill")
    return True

