import socket  # for sockets
import sys  # for exit
import os  # for
import time  # for sleep
import json  # for decode json
import errno  # for echo exception

import numpy as np
from handle import Xgb_clf as clf

import command as Cmd
from handle_nn import NN_clf


class base:  # get data form ms_tts_server and translate to the jiajia robot's mouth synthesis message
    def __init__(self, ip, port):  # init get ip and port from parent thread
        self.state = None
        self.s = None  # socket
        self.m = None  # message
        self.req = None  # request
        self.conn = None
        self.data = None  # date from server
        self.data_ok = None

        self.my_ip = str(ip)  # client ip
        self.my_port = int(port)  # client port
        self.my_addr = (self.my_ip, self.my_port)
        self.open_socket()  # open socket
        while 1:
            self.start_service()  # get server replay
            # self.pub_ip.publish(String(self.my_ip))

    def open_socket(self, args=None):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # open socket
        except (socket.error) as msg:
            print('Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + str(
                msg[1]))  # print open socket error message
            sys.exit()  # exit
        try:
            self.s.bind(self.my_addr)  # bind client and port
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + str(msg[1]))  # print bind error message
            sys.exit()  # exit
        print('Socket bind complete')
        self.s.listen(5)  # listen the port
        print('Socket now listening')

    def start_service(self, args=None):
        print('starting BASE')
        self.conn, addr = self.s.accept()  # receiving data
        # data=""
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
        # publish ros message
        while True:  # loop
            try:
                data = self.conn.recv(4096)  # received data ,buffer size
                self.data = data
                self.handle()
                break
            except socket.error as e:
                if e.args[0] == errno.EWOULDBLOCK:  # dealing socket error messages
                    break
                print (e)

    def handle(self, args=None):
        if len(self.data) > 0:
            try:
                d = self.data.decode('utf-8')
                print('receive data ok')
                features = d.split('\t')
                har = self.inference1(features)
                self.conn.send(str(har).encode('utf-8'))
            except Exception as e:
                print('bad line')  # print error message
                print('Failed to create socket. Error code: ' + str(e))

    def inference(self,features):
        inf = clf()
        inf.module_load()
        print("begin")
        res = inf.inference(features)
        har = (int)(res[0])
        print('state: ',har)
        return har

    def inference1(self,features):
        print(features)
        inf = NN_clf()
        print("begin")
        res = inf.inference(features)
        print(res)
        har = (int)(res[0])
        print('state: ',har)
        return har



    def close_socket(self, args=None):  # close socket
        self.s.close()
