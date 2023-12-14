import cv2
import mediapipe as mp
import numpy as np
import time

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
mp_pose = mp.solutions.pose
ATTRIBUTES = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER] #,mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

class bgMask:
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)
        self.flowWindow = 10
        self.transforms = [] 
        self.prevGrey = None
        self.currentBackground = cv2.imread("backgrounds/pexels-steve-chai-5204175.jpg")
        self.currentBackground = cv2.resize(self.currentBackground, (192,256))

    def run(self, image):
        # results = self.faceModel.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return 
        attVisibility = []
        for i in ATTRIBUTES:
            attVisibility.append(results.pose_landmarks.landmark[i].visibility)
        if min(attVisibility) < 0.9:
            return 
        
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.05
        bgImamage = np.zeros(image.shape, dtype=np.uint8)
        bgImamage[:] = MASK_COLOR
        output_image = np.where(condition, image[:, :, ::-1], bgImamage)

        self.BGcondition = np.stack((results.segmentation_mask,) * 3, axis=-1) <= 0.9
        bg = np.where(self.BGcondition, image, MASK_COLOR)
        bg = np.where(self.BGcondition, image, bg)
        self.currentBackground = bg
        return output_image
    
    def retrieveBG(self, image):
        outputImage = np.where(self.BGcondition, self.currentBackground , image)
        return outputImage
        

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size)/window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed
    
    def smooth(self, trajectory):
        SMOOTHING_RADIUS = 1
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
            smoothed_trajectory[:,i] = self.movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
        return smoothed_trajectory
    
    def fixBorder(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def getOutputImage(self, currentImage):
        currentImage = np.uint8(currentImage)
        self.prevGrey = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY) if  type(self.prevGrey) != np.ndarray else self.prevGrey
        prevPts = cv2.goodFeaturesToTrack(self.prevGrey,maxCorners=10,qualityLevel=0.01,minDistance=30,blockSize=3)
        currentGrey = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY) 
        currPts, status, err = cv2.calcOpticalFlowPyrLK(self.prevGrey, currentGrey, prevPts, None)
        m,_ = cv2.estimateAffine2D(prevPts, currPts) #will only work with OpenCV-3 or less
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0])
        if len(self.transforms)!=0:
            self.transforms = np.vstack([self.transforms,[dx,dy,da]])
        else:
            self.transforms = np.array([[dx,dy,da]])
        if self.transforms.shape[0] <= self.flowWindow:
            self.prevGrey  = currentGrey
            # print(self.transforms)
            return currentImage
        self.transforms = np.delete(self.transforms, obj = 0, axis = 0)
        trajectory = np.cumsum(self.transforms, axis=0)
        smoothed_trajectory = self.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        transforms_smooth = self.transforms + difference
        dx = transforms_smooth[4,0]
        dy = transforms_smooth[4,1]
        da = transforms_smooth[4,2]
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        frame_stabilized = cv2.warpAffine(currentImage, m, (192,256))
        frame_stabilized = self.fixBorder(frame_stabilized)
        self.prevGrey  = currentGrey
        # TODO: fix border if necessary
        return frame_stabilized


