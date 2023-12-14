import cv2
import numpy as np
import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import openrtist_pb2
import os
from io import BytesIO
from maskBG import bgMask
from PIL import Image
logger = logging.getLogger(__name__)

import json

import sys
from clothIndex import getCloth, getBG

sys.path.append('DAFlow/')
from preinference import DAFLOW

class DressUpEngine(cognitive_engine.Engine):
    SOURCE_NAME = "openrtist"

    def __init__(self, compression_params):
        self.compression_params = compression_params
        # self.overlay = Image.open("noPoseB.png")
        # self.overlay = self.overlay.convert("RGBA")
        # self.overlay = cv2.resize(self.overlay,(192,256))
        # wtr_mrk4 = cv2.imread("../wtrMrk.png", -1)
        # self.mrk, _, _, mrk_alpha = cv2.split(wtr_mrk4)
        # self.alpha = mrk_alpha.astype(float) / 255
        logger.info("FINISHED-INITIALISATION")
        # self.obj = dressUpInference()
        self.mask = bgMask()
        # self.obj = DAFLOW()
    
    def handle(self, input_frame):
        extras = cognitive_engine.unpack_extras(openrtist_pb2.Extras, input_frame)
        # clothId = json.loads(extras.style)['clothID']
        # clothId,edgeId = getCloth(clothId)
        clothId,edgeId = getCloth("0")
        bgId = json.loads(extras.style).get("bgID")
        bgPath = getBG(bgId) if bgId != '0' else bgId
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        orig_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        orig_img = cv2.resize(orig_img, (192,256))
        #image = orig_img
        # image = self.mask.run(orig_img)
        # image = self.obj.run(orig_img, clothPath = clothId, bg = bgPath)
        # print(bgPath)
        image = self.obj.run(orig_img, bg = bgPath)
        # if image is not None:
        #     # image = self.obj.run(image, bg = bgPath) # , clothId)
        #     # image = self.mask.retrieveBG(image)
        #     pass
        # else:
        #     # image = Image.fromarray(orig_img)
        #     # image = self.applyPrompt(image)
        #     # image = np.array(orig_img)
        #     image = orig_img

        # image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGBA2RGB)

        img_data = orig_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        _, jpeg_img = cv2.imencode(".jpg", image, self.compression_params)
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
    
    def _apply_watermark(self, image):
        img_mrk = image[-30:, -120:]  # The waterMark is of dimension 30x120
        img_mrk[:, :, 0] = (1 - self.alpha) * img_mrk[:, :, 0] + self.alpha * self.mrk
        img_mrk[:, :, 1] = (1 - self.alpha) * img_mrk[:, :, 1] + self.alpha * self.mrk
        img_mrk[:, :, 2] = (1 - self.alpha) * img_mrk[:, :, 2] + self.alpha * self.mrk
        image[-30:, -120:] = img_mrk
        img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return img_out

    def applyPrompt(self, image):
        # return cv2.addWeighted(image, 0.4, self.overlay, 0.1, 0) 
        image.paste(self.overlay, (0, 0), self.overlay)
        return image

    
