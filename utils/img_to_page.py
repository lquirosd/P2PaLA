from __future__ import print_function
from __future__ import division
from builtins import range

import logging
import sys
import os
import glob

import re
import numpy as np
import cv2
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from page_xml.xmlPAGE import pageData
import matplotlib.pyplot as plt


def main():
    """Build a PAGE-XML file from img encoded data"""
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    nm = re.split(r"-", sys.argv[3])
    name_map = {}
    for k in nm:
        z, c = re.split(r":", k)
        name_map[z] = tuple(c.split(","))

    name_map = {"marginalia": [0, 0, 255]}
    formats = ["tif", "tiff", "png", "jpg", "jpeg", "JPG", "bmp"]
    img_paths = []
    for ext in formats:
        img_paths.extend(glob.glob(in_path + "/*." + ext))
    img_ids = [os.path.splitext(os.path.basename(x))[0] for x in img_paths]
    img_data = dict(zip(img_ids, img_paths))

    for img_id, img_p in img_data.items():
        page = pageData(os.path.join(out_path, img_id + ".xml"), logger=None)
        img_name = os.path.basename(img_p)
        # --- open file
        img = cv2.imread(img_p)
        rows, cols, _ = img.shape
        page.new_page(img_name, str(rows), str(cols))
        r_id = 0
        for zone_name, color in name_map.items():
            zones_img = np.all(img == color, axis=-1).astype(np.uint8)
            _, contours, hierarchy = cv2.findContours(
                zones_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                reg_coords = ""
                for x in cnt.reshape(-1, 2):
                    reg_coords = reg_coords + " {},{}".format(x[0], x[1])
                text_reg = page.add_element(
                    "TextRegion", str(r_id), zone_name, reg_coords.strip()
                )
                r_id += 1
        page.save_xml()


if __name__ == "__main__":
    main()
