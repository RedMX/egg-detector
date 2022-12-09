# egg-detector
a python AI that uses tensorflow's object detection API and zeroMQ for communication to detect eggs in images using a custom trained model

Requirements:

opencv-python,pyzmq,tensorflow(2.10.1),object_detection(0.1),protobuf(3.20.3)
you can install everything except object_detection from pip you have to install object_detection from tensorflow's repo
you can do it with these commands(tested on linux):
git clone https://github.com/tensorflow/models

cd ./models/research

protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .

Example Usage:

egg_ai_threshold='0.8' python ./egg_detector.py
then in another terminal window
python ./client.py

