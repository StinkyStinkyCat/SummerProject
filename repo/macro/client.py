import socket
import time


def time_to_string():
    return time.strftime("%d/%m/%Y  %H:%M:%S   ", time.localtime())


def print_with_time(message: str):
    print(time_to_string() + message)


buffer_size = 32


class Client:
    def __init__(self, ip='localhost', port=8080):
        print_with_time("Initializing server...")
        self.ip = ip
        self.port = port
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.__handshake()

    def __handshake(self):
        res = self.__send_message("MACRO")
        if res != "BLENDER":
            print_with_time("Unexpected message: \"" + res + "\"; Expected: \"BLENDER\".")
            print_with_time("Failed handshake.")
            exit(1)
        print_with_time("Successful handshake.")

    def __send_message(self, message: str):
        self.soc.sendto(message.encode(), (self.ip, self.port))
        print_with_time("Message sent: \"" + message + "\".")
        return self.soc.recv(buffer_size).decode()

    def wait_for_finished_message(self):
        print_with_time("Waiting for the \"finished\" message...")
        res = self.soc.recv(buffer_size).decode()
        while res != 'finished':
            print_with_time("Unexpected message: \"" + res + "\"; Expected: \"finished\".")
            res = self.soc.recv(buffer_size).decode()
        print_with_time("Message \"finished\" received.")

    def send_render_request(self):
        return self.__send_message("render")

    def send_stop_request(self):
        return self.__send_message("stop")
