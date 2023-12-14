#!/usr/bin/env python3

from gabriel_server import local_engine
from lookUpEngine import lookUpEngine
from timing_engine import TimingEngine
import logging
import cv2
import argparse
import importlib
import dill as pickle
import torch
import torch.nn as nn

DEFAULT_PORT = 9098
DEFAULT_NUM_TOKENS = 2
INPUT_QUEUE_MAXSIZE = 60
DEFAULT_STYLE = "the_scream"
COMPRESSION_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 67]

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

import sys
# sys.path.append('/home/arexhari/aylmer843/hugFace/search_similar_image')
# from search_similar_image.model import SearchModel
from searchInference import searchInference





def create_adapter(openvino, cpu_only, force_torch, use_myriad):
    """Create the best adapter based on constraints passed as CLI arguments."""
    return TorchAdapter(False, DEFAULT_STYLE)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t", "--tokens", type=int, default= DEFAULT_NUM_TOKENS, help="number of tokens"
    )
    parser.add_argument(
        "-o",
        "--openvino",
        action="store_true",
        help="Pass this flag to force the use of OpenVINO."
        "Otherwise Torch may be used",
    )
    parser.add_argument(
        "-c",
        "--cpu-only",
        action="store_true",
        help="Pass this flag to prevent the GPU from being used.",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Set this flag to force the use of torch. Otherwise"
        "OpenVINO may be used.",
    )
    parser.add_argument(
        "--myriad",
        action="store_true",
        help="Set this flag to use Myriad VPU (implies use OpenVino).",
    )
    parser.add_argument(
        "--timing", action="store_true", help="Print timing information"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=DEFAULT_PORT, help="Set port number"
    )
    args = parser.parse_args()

    def engine_setup():
        adapter = create_adapter(args.openvino, args.cpu_only, args.torch, args.myriad)
        if args.timing:
            engine = TimingEngine(COMPRESSION_PARAMS, adapter)
        else:
            engine = lookUpEngine(COMPRESSION_PARAMS, adapter)

        return engine
    #create_adapter(args.openvino, args.cpu_only, args.torch, args.myriad)
    logger.info(str(args.port)+"is the port number")
    local_engine.run(
        lambda: lookUpEngine(COMPRESSION_PARAMS),
        lookUpEngine.SOURCE_NAME,
        INPUT_QUEUE_MAXSIZE,
        args.port,
        args.tokens,
    )


if __name__ == "__main__":
    main()
