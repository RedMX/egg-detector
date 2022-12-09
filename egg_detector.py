
import os
import zmq
import tensorflow as tf
import cv2
import json
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from urllib.request import Request, urlopen
category_index = label_map_util.create_category_index_from_labelmap('./label_map.pbtxt')
config=config_util.get_configs_from_pipeline_file('./model/pipeline.config')
model=model_builder.build(model_config=config["model"],is_training=False)
ckpt=tf.compat.v2.train.Checkpoint(model=model)
ckpt.restore('./model/ckpt-26').expect_partial()
def detect_fn(image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections




def img_chck(url):


    req = Request(
        url=url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
    detections['num_detections'] = num_detections


    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)



    im_height,im_width,dim=img.shape
    for i in range(len(detections['detection_scores'])):
        if detections['detection_scores'][i]>float(os.environ['egg_ai_threshold']):
            img = cv2.rectangle(img,(int(detections['detection_boxes'][i][1]*im_width),int(detections['detection_boxes'][i][0]*im_height)) ,(int(detections['detection_boxes'][i][3]*im_width),int(detections['detection_boxes'][i][2]*im_height)), (0,0,255), 5)

            #img = cv2.putText(img, category_index[detections['detection_classes'][i]+1]['name']+": "+str(np.around(detections['detection_scores'][i]*100,2))+"%",(int(detections['detection_boxes'][i][1]*im_width),int(detections['detection_boxes'][i][0]*im_height)-30),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5,cv2.LINE_AA) uncomment for text(may not work well with lower resolution images)
    if np.max(detections['detection_scores']) > float(os.environ['egg_ai_threshold']):
        cv2.imwrite('./eggs.jpg',img)
        return str(np.around(np.max(detections['detection_scores'])*100,2))
    return 0

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
while True:
 
    message = socket.recv()
    message=json.loads(message.decode("utf-8"))
    socket.send(str(img_chck(message['url'])).encode(encoding= 'UTF-8'))

   
