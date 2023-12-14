import cv2
import numpy as np

import sys
sys.path.append('./PF-AFN_test')
from maskBG import bgMask
from faceMask import bgMask as fm
from inference import dressUpInference


sys.path.append('DAFlow/')
from videoHelper import DAFLOW


dafObject = DAFLOW()
afnObject = dressUpInference()
mask = bgMask()

clothId = ""
clothPath = ""

cap = cv2.VideoCapture("/home/arexhari/videos/trial1.mp4")
index = 0

while(cap.isOpened()):
    # print("converting")
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (192, 256))

        #run PF-AFN
        afnResult = frame
        afnFace = frame
        image = mask.run(frame)

        if image is not None:
            image = afnObject.runInference(image, "0")
            afnResult = mask.retrieveBG(image)
        
        #run DAF
        dafResult = dafObject.run(frame, clothPath = "/home/arexhari/dressup/Self-Correction-Human-Parsing/output/sim.png", bg = '0' ) #"backgrounds/pexels-steve-chai-5204175.jpg"
        if dafResult is None:
            dafResult = frame
        
        clothImage = cv2.imread("/home/arexhari/dressup/Self-Correction-Human-Parsing/output/sim.png")
        vis = np.concatenate((clothImage, dafResult), axis=1) #afnResult,
        
        # vis = dafResult
        cv2.imwrite("videoResults/%d.jpg"%(index), vis)
        # out.write(vis)
        index+=1
    else:
        break

cap.release()
cv2.destroyAllWindows()
# /home/arexhari/dressup/Self-Correction-Human-Parsing/output/sim.png
print("The video was successfully saved")