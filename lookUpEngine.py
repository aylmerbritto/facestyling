import cv2
import numpy as np
from PIL import Image

from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2

import logging
from io import BytesIO
from maskBG import bgMask

import json
import sys
import openrtist_pb2

logger = logging.getLogger(__name__)
from searchInference import searchInference
import sys

class lookUpEngine(cognitive_engine.Engine):
    SOURCE_NAME = "openrtist"
    def __init__(self, compression_params):
        self.compression_params = compression_params
        logger.info("FINISHED-INITIALISATION")
        self.obj = searchInference()
    
    def handle(self, input_frame):
        extras = cognitive_engine.unpack_extras(openrtist_pb2.Extras, input_frame)
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        orig_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        orig_img = Image.fromarray(orig_img).convert("RGB")
        _, orig_img = self.obj.run(orig_img)
        # orig_img = np.array(orig_img)
        img_data = orig_img.tostring()
        result = gabriel_pb2.ResultWrapper.Result() 
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        _, jpeg_img = cv2.imencode(".jpg", orig_img, self.compression_params)
        img_data = jpeg_img.tostring()
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        extras = openrtist_pb2.Extras()
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)
        result_wrapper.extras.Pack(extras)
        return result_wrapper

