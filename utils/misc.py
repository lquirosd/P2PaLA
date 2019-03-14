#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from builtins import range

import sys
import os
import glob


def check_input_folder(pointer, check_xml=True):
    """
    check is input folder contains images and the XML file
    """
    formats = ["tif", "tiff", "png", "jpg", "jpeg", "JPG", "bmp"]
    #--- check if contains images
    img_paths = []
    error_msg=""
    for ext in formats:
        img_paths.extend(glob.glob(pointer + "/*." + ext))
    if len(img_paths) == 0:
        error_msg += "No image files found on folder {}.".format(pointer)
        return False, error_msg
    #--- check XML file
    if check_xml:
        for f in img_paths:
            name = os.path.splitext(os.path.basename(f))[0]
            dir = os.path.dirname(f) 
            xml_path = os.path.join(dir, 'page', name+'.xml')
            if not (os.path.isfile(xml_path) and os.access(xml_path, os.R_OK)):
                error_msg += "No XML file found for file {}\n".format(f)
        if error_msg == "":
            return True, error_msg
        else:
            return False, error_msg
    else:
        return True, error_msg
     
