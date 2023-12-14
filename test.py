import cv2
from DAFlow.prep import bgMask
import numpy as np
import DAFlow.pose as pose

import sys
sys.path.append("/home/arexhari/dressup/Self-Correction-Human-Parsing/")
import parserAgn

imPath = '/home/arexhari/dressup/Self-Correction-Human-Parsing/single-demo/015794_0.jpg'
inputImage = cv2.imread(imPath)

f1 = bgMask()
hParser = parserAgn.humanParser()

preOutput = f1.run(inputImage)
cleanImage,poseData = preOutput
agnosticRGB = hParser.run(cleanImage)
agnosticRGB = np.uint8(agnosticRGB)
print(type(poseData))
_, poseRGB, agnosticRGB = pose.getPose(cleanImage, poseData, agnosticRGB)
agnosticRGB.save("results/agn.png")