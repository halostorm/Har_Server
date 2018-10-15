import socket  # for sockets
import fcntl
import struct
import threading  # for thread
import random  # for random
import re
import subprocess
from base import base  # for base thread


def Base(ip, port):  # client thread
    b = base(ip, port)


def find_all_ip():
    ipstr = '([0-9]{1,3}\.){3}[0-9]{1,3}'
    ipconfig = subprocess.Popen("ifconfig", stdout=subprocess.PIPE)
    output = ipconfig.stdout.read()
    ip_pattern = re.compile('(inet addr:%s)' % ipstr)
    pattern = re.compile(ipstr)
    iplist = []
    for ipaddr in re.finditer(ip_pattern, str(output)):
        ip = pattern.search(ipaddr.group())
        if ip.group() != '127.0.0.1' and ip.group() != '192.168.56.1':
            iplist.append(ip.group())
    return iplist


if __name__ == '__main__':
    iplist = []
    ip = None
    iplist = find_all_ip()

    print ('find IPs to use:')

    if iplist == []:
        ip = 'localhost'
    else:
        ip = iplist[0]
    print ('Starting Server at IP:')
    print(ip)

    # print 'publish data'

    threads = []

    t1 = threading.Thread(target=Base, args=(ip, 10086))
    # t2 = threading.Thread(target=Exp,args=(ip,10088))
    # t3 = threading.Thread(target=EML,args=(ip,10089))
    # t4 = threading.Thread(target=say_ip,args=(ip))
    threads.append(t1)
    # threads.append(t2)
    # threads.append(t3)
    # threads.append(t4)
    for t in threads:  # start threads
        t.setDaemon(True)
        t.start()
    t.join()
