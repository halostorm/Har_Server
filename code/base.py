import socket  # for sockets
import sys  # for exit
import os  # for
import time  # for sleep
import json  # for decode json
import errno  # for echo exception
import command as Cmd

class base:  # get data form ms_tts_server and translate to the jiajia robot's mouth synthesis message
    def __init__(self, ip, port):  # init get ip and port from parent thread
        self.state = None
        self.s = None  # socket
        self.m = None  # message
        self.req = None  # request
        self.conn = None
        self.data = None  # date from server
        self.data_OK = False  # server replay messages finished

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
            print ('Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + str(msg[1]))  # print open socket error message
            sys.exit()  # exit
        try:
            self.s.bind(self.my_addr)  # bind client and port
        except socket.error as msg:
            print ('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + str(msg[1]))  # print bind error message
            sys.exit()  # exit
        print ('Socket bind complete')
        self.s.listen(5)  # listen the port
        print ('Socket now listening')


    def start_service(self, args=None):
        print ('starting BASE')
        self.conn, addr = self.s.accept()  # receiving data
        # data=""
        print ('Connected with ' + addr[0] + ':' + str(addr[1]))
        # publish ros message tts_syth
        while True:  # loop
            try:
                data = self.conn.recv(4096)  # received data ,buffer size 4096
            except socket.error as e:
                if e.args[0] == errno.EWOULDBLOCK:  # dealing socket error messages
                    break
                # raise
                print (e)
                data = None
            else:
                if not data:
                    break
                else:
                    self.data = data
                    self.data_OK = True
                    self.get_result()  # decode base message
                    # res,rep=self.get_result()
                    # print rep , res
                    # if res!=None:
                    # self.pub_cmd.publish(res)	#publish  ros message
                    # self.conn.send(rep)

    def get_result(self, args=None):
        if len(self.data) > 0 and self.data_OK == True:
            try:
                d = json.loads(self.data.decode('utf-8'))  # decode json structure
                print(d)
                self.data_OK = False
                if d["type"] == "request":
                    if d["service"] == "connect":  # return first base data
                        result = Cmd.run()
                        rep = self.gen_reply("connect", result).encode('utf-8')
                        self.conn.send(rep)
                    elif d["service"] == "start":
                        self.state = "start"
                        fpsValue = d["value"]
                        result = Cmd.send_event(2,1,fpsValue)
                        rep = self.gen_reply("start", result).encode('utf-8')
                        self.conn.send(rep)
                    elif d["service"] == "stop":
                        self.state = "stop"
                        fpsValue = d["value"]
                        result = Cmd.send_event(2,2,fpsValue)
                        rep = self.gen_reply("stop", result).encode('utf-8')
                        self.conn.send(rep)
                    elif d["service"] == "disconnect":
                        result = Cmd.kill()
                        rep = self.gen_reply("disconnect", result).encode('utf-8')
                        self.conn.send(rep)
                    elif d["service"] == "changeFps":
                        fpsValue = d["value"]
                        if self.state == "start":
                            result = Cmd.send_event(2, 1,fpsValue)
                        else :
                            result = Cmd.send_event(2, 2, fpsValue)

                        rep = self.gen_reply("changeFps", result).encode('utf-8')
                        self.conn.send(rep)
                    else:
                        print ("No match")
                        rep = self.gen_reply("no_match", False).encode('utf-8')
                        self.conn.send(rep)
                else:
                    print ("No match")
                    rep = self.gen_reply(0, False).encode('utf-8')
                    self.conn.send(rep)
            except Exception as e:
                print ('bad line')  # print error message
                print ('Failed to create socket. Error code: ' + str(e))


    def gen_reply(self, typ, status):
        if status == True:
            s = "true"
        else:
            s = "false"
        t = str(time.time())
        data = "{\n" + \
               "\"type\":\"reply\",\n" + \
               "\"time\":\"" + t + "\",\n" + \
               "\"service\":\"" + typ + "\",\n" + \
               "\"result\":" + s+" }"
        return data

    def close_socket(self, args=None):  # close socket
        self.s.close()

