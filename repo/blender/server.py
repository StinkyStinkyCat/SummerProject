import render
import socket
import time

buffer_size = 32


def time_to_string():
    return time.strftime("%d/%m/%Y  %H:%M:%S   ", time.localtime())


def log(string: str):
    print(time_to_string() + string)


class PuppyServer:
    client_address_port = ('', 0)
    count = 0

    def __init__(self, client_ip='127.0.0.1', port=8080):
        self.client_ip = client_ip
        self.port = port
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.soc.bind((client_ip, port))
        self.__handshake()
        self.renderer = render.BlenderController()

    def __handshake(self):
        log("Waiting for connection...")
        inf = self.__check_client_ip()
        if inf[0].decode() != "MACRO":
            log("Unexpected message: \"" + inf[0].decode() + "\"; Expected: \"MACRO\".")
            self.__handshake()
        else:
            log("Successful handshake.")
            self.client_address_port = inf[1]
            self.__send_message('BLENDER')

    def __check_client_ip(self):
        inf = self.soc.recvfrom(buffer_size)
        if inf[1][0] != self.client_ip:
            self.soc.sendto("go away!".encode(), inf[1])
            log("Unknown connection!")
            log("Message: {}".format(inf[0]))
            log("Address: {}".format(inf[1]))
            exit(1)
        return inf

    def __receive_message(self):
        inf = self.__check_client_ip()
        return inf[0].decode()

    def __send_message(self, message: str):
        self.soc.sendto(message.encode(), self.client_address_port)

    def handle(self):
        request = ''
        while request != 'stop':
            request = self.__receive_message()
            if request == 'render':
                log('Request received...')
                self.__send_message('ok')
                # render now !!!
                log('Rendering...')
                self.renderer.render_all(str(self.count + 1).zfill(3))
                self.count += 1
                # return finished message
                self.__send_message('finished')
            elif request != 'stop':
                log("Unexpected message: \"" + request + "\"; Expected: \"render\" or \"stop\".")
        # deal with the stop request
        self.__send_message('ok')
        log('Closing server...')
        log('Totally rendered ' + str(self.count) + ' sets.')
