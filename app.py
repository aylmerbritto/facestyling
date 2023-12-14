import cv2
import logging

from gabriel_server import local_engine
from stylingEngine import stylingEngine

DEFAULT_PORT = 9099
DEFAULT_NUM_TOKENS = 2
INPUT_QUEUE_MAXSIZE = 60
COMPRESSION_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 67]

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    def engine_factory():
        return stylingEngine(COMPRESSION_PARAMS)
    
    local_engine.run(lambda: stylingEngine(COMPRESSION_PARAMS),'openrtist',INPUT_QUEUE_MAXSIZE, DEFAULT_PORT, DEFAULT_NUM_TOKENS)

if __name__ == "__main__":
    main()