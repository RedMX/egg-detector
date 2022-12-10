import zmq
import json
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
url=input("enter image url: ")
socket.send(str(url).encode(encoding = 'UTF-8'))
message = socket.recv()
print(message) # returns json formatted message with the highest percentage out of all detections and imgur link or 0 if there is no detection above set threshold
