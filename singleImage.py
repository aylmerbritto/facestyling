import sys
sys.path.append('/home/arexhari/aylmer843/openrtist/server/DAFlow')
from DAFlow.pose import getPose
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import DAFlow.test_SDAFNet_viton as inf
import torch
from DAFlow.models.sdafnet import SDAFNet_Tryon

from DAFlow.prep import bgMask
import cv2



def imageUtil(image):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    image = transform(image).unsqueeze(0)
    image = image.cuda()
    return image


class singleImage:
    def __init__(self) -> None:
        self.f1 = bgMask()
        opt = inf.get_opt()
        self.sdafnet = SDAFNet_Tryon(ref_in_channel=opt.multi_flows).cuda()
        self.sdafnet = self.sdafnet.cuda()
        self.sdafnet.load_state_dict(torch.load("DAFlow/ckpt_viton.pt"))
        self.sdafnet.eval()

    def run(self, image, cloth, pose, agnostics):
        inputs = {
            'img_name': imagePath,
            'c_name': {'paired':clothPath},
            'img': image,
            'img_agnostic': agnostics,
            'pose': pose,
            'cloth': {'paired':cloth},
        }
        output = inf.inference(inputs, self.sdafnet)
        output = np.array(output)
        return output

imagePath = "/home/arexhari/dressup/Self-Correction-Human-Parsing/single-demo/015794_0.jpg"
clothPath = "PF-AFN_test/dataset/test_clothes/019119_1.jpg"
agnosticsPath = "/home/arexhari/dressup/Self-Correction-Human-Parsing/output/sim.png"

f1 = bgMask()
image = cv2.imread(imagePath)
cloth = cv2.imread(clothPath)
agnostics = cv2.imread(agnosticsPath)
cleanImage,poseData = f1.run(image)
_, poseImage, agnostics = getPose(cleanImage,poseData)
poseD = poseImage
agD = agnostics
imageD = image
clothD = cloth

image = imageUtil(cleanImage)
cloth = imageUtil(cloth)
agnostics = imageUtil(agnostics)
poseImage = imageUtil(poseImage)

obj = singleImage()
output = obj.run(image, cloth, poseImage, agnostics)
output = np.concatenate((imageD, clothD, poseD, agD, output),axis=1)
cv2.imwrite("results/agnosticsTest.png", output)
