from __future__ import print_function
from __future__ import division
from builtins import range

import logging
import errno
import sys
import os

from collections import OrderedDict
import torch


def get_model(model_in, model_out):
    """Remove unnecessary data from checkpoint"""
    checkpoint = torch.load(model_in, map_location=lambda storage, loc: storage)
    out_file = os.path.join(model_out, "P2PaLA_inferenceModel.pth")
    torch.save({"nnG_state": checkpoint["nnG_state"]}, out_file)


def main():
    """
    """
    model_in = sys.argv[1]
    if not (os.path.isfile(model_in) and os.access(model_in, os.R_OK)):
        raise FileNotFoundError(model_in)
    model_out = sys.argv[2]
    if not (os.path.isdir(model_out) and os.access(model_out, os.R_OK)):
        raise FileNotFoundError(model_out)

    get_model(model_in, model_out)
    print(
        "Done: model saved to {}".format(
            os.path.join(model_out, "P2PaLA_inferenceModel.pth")
        )
    )


if __name__ == "__main__":
    main()
