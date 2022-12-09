import zmq
import json
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
url=input("enter image url: ")
x={
"url": url
}
socket.send(str(json.dumps(x)).encode(encoding = 'UTF-8'))
message = socket.recv()
print(message) # returns the highest percentage out of all detections or 0 if there is no detection above set threshold
