import time
import os
import tempfile
from PIL import Image
import cv2

import sys
sys.path.append('search_similar_image/')
from search_similar_image.model import SearchModel

REFERENCE_IMAGES = 'search_similar_image/index'
# REFERENCE_IMAGES = "/home/ubuntu/data/index"
class searchInference:
    def __init__(self) -> None:
        self.obj = SearchModel()
        self.obj.fit(REFERENCE_IMAGES)
        self.linkerName = "/home/ubuntu/data/currentShirt.jpg"
        # self.linkerName = "currentShirt.jpg"
    
    def symlink_force(self, target, link_name):
        '''
        Create a symbolic link link_name pointing to target.
        Overwrites link_name if it exists.
        '''

        # os.replace() may fail if files are on different filesystems
        link_dir = os.path.dirname(link_name)

        while True:
            temp_link_name = tempfile.mktemp(dir=link_dir)
            try:
                os.symlink(target, temp_link_name)
                break
            except FileExistsError:
                pass
        try:
            os.replace(temp_link_name, link_name)
        except OSError:  # e.g. permission denied
            os.remove(temp_link_name)
            raise

    def run(self, image = None):
        scores, similar_image_paths = self.obj.predict(image)
        self.symlink_force(similar_image_paths[0][0], self.linkerName)
        return scores, cv2.imread(self.linkerName)


if __name__ == '__main__':
    model = searchInference()
    img = Image.open("/home/arexhari/aylmer843/hugFace/search_similar_image/j.png").convert('RGB')
    for i in range(10):
        startTime = time.time()
        model.run(img)
        endTime = time.time()
        print(endTime-startTime)