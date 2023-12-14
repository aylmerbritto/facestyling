import numpy as np
from PIL import Image
import cv2
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import logging
logger = logging.getLogger(__name__)

import sys
sys.path.append('DualStyleGAN/')
import faceStyling_pb2


class stylingEngine(cognitive_engine.Engine):
    SOURCE_NAME = 'caricature'
    def __init__(self, compression_params=None) -> None:
        self.compression_params = compression_params
        from inference_script import inferenceStyle
        self.obj = inferenceStyle()
        logger.info("initialised Model")

    def handle(self, input_image):
        extras = cognitive_engine.unpack_extras(faceStyling_pb2.Extras, input_image)
        np_data = np.frombuffer(input_image.payloads[0], dtype=np.uint8)
        frame_rgb = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        frame_rgb = cv2.resize(frame_rgb, (240*3, 320*3))
        # print(frame_rgb.shape)
        # cv2.imwrite('input.jpeg', frame_rgb)
        pil_image = Image.fromarray(frame_rgb)
        image = self.obj.handle(pil_image)
        resized_image = cv2.resize(image, None, fx=0.5, fy=0.5)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        _, jpeg_img = cv2.imencode(".jpg", resized_image, self.compression_params)

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = jpeg_img.tostring()
        extras = faceStyling_pb2.Extras()

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)
        result_wrapper.extras.Pack(extras)
        return result_wrapper