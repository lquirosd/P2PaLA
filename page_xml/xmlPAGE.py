from __future__ import print_function
from __future__ import division

import os
import logging

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import re


class pageData():
    """ Class to process PAGE xml files"""
    def __init__(self,filepath):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.parse()

    def parse(self):
        """
        Parse PAGE-XML file
        """
        tree = ET.parse(self.filepath)
        #--- get the root of the data
        self.root = tree.getroot()
        #--- save "namespace" base
        self.base = self.root.tag.rsplit('}',1)[0] + '}'
    
    def getRegion(self,region_name):
        """
        get all regions in PAGE which match region_name
        """
        return self.root.findall(".//"+self.base+region_name) or None

    def getId(self,element):
        """
        get Id of current element
        """
        return str(element.attrib.get('id'))

    
    def getRegionType(self, element):
        """
        Returns the type of element
        """
        try:
            e_type =  re.match(r".*structure {.*type:(.*);.*}",
                      element.attrib['custom']).group(1)
        except:
            e_type = None
            logging.warning('No region type defined for {} at {}'.format(
                            self.getId(element),
                            self.name))
        return e_type

    def getSize(self):
        """
        Get Image size defined on XML file
        """
        img_width = int(self.root.findall("./"+self.base+"Page")[0].get(
                        'imageWidth'))
        img_height = int(self.root.findall("./"+self.base+"Page")[0].get(
                        'imageHeight'))
        return (img_width, img_height)

    def getCoords(self, element):
        str_coords = element.findall('./' + self.base +
                                    'Coords')[0].attrib.get('points').split()
        return np.array([i.split(',') for i in str_coords]).astype(np.int)


    def buildMask(self,out_size,element_name, color_dic):
        """
        Builds a "image" mask of desired elements
        """
        size = self.getSize()[::-1]
        mask = np.zeros(out_size, np.uint8) + 255
        scale_factor = out_size/size
        #mask = np.zeros((int(size[1]*scale_factor),
        #                 int(size[0]*scale_factor)), np.uint8) + 255
        for element in self.root.findall(".//"+self.base+element_name):
            #--- get element type
            e_type = self.getRegionType(element)
            if (e_type == None or e_type not in color_dic):
                e_color = 175
                logging.warning('Element type "{}"undefined on color dic, set to default=175'.format(e_type))
            else:
                e_color = color_dic[e_type]
            #--- get element coords
            coords = self.getCoords(element)
            coords = (coords * np.flip(scale_factor,0)).astype(np.int)
            cv2.fillConvexPoly(mask,coords,e_color)
        return mask

    def buildBaselineMask(self,out_size,color,line_width):
        """
        Builds a "image" mask of Baselines on XML-PAGE
        """
        size = self.getSize()[::-1]
        mask = np.zeros((out_size[0],out_size[1]), np.uint8) + 255
        scale_factor = out_size/size
        #mask = np.zeros((int(size[1]*scale_factor),
        #                 int(size[0]*scale_factor)), np.uint8) + 255
        for element in self.root.findall(".//"+self.base+'Baseline'):
            #--- get element coords
            str_coords = element.attrib.get('points').split()
            coords = np.array([i.split(',') for i in str_coords]).astype(np.int)
            coords = (coords * np.flip(scale_factor,0)).astype(np.int)
            cv2.polylines(mask, [coords.reshape(-1,1,2)], False, color, line_width)
        return mask
    
    def getText(self,element):
        """
        get Text defined for element
        """
        text_node = element.find('./' + self.base + 'TextEquiv')
        if (text_node == None):
            logging.warning('No Text node found for line {} at {}'.format(
                            self.getId(element),
                            self.name))
            return ''
        else:
            text_data = text_node.find('*').text
            if (text_data == None):
                logging.warning('No text found in line {} at {}'.format(
                                self.getId(element),
                                self.name))
                return ''
            else:
                return text_data.encode('utf-8').strip()

    def getTranscription(self):
        """
        Extracts text from ech line on the XML file
        """
        data = {}
        for element in self.root.findall(".//"+self.base+'TextRegion'):
            r_id = self.getId(element)
            for line in element.findall(".//"+self.base+'TextLine'):
                l_id = self.getId(line)
                data[r_id + '_' + l_id] = self.getText(line)

        return data

    def writeTranscriptions(self,out_dir):
        """
        write out one file per line
        """
        for line,text in self.getTranscription().iteritems():
            fh = open (out_dir + '/' + self.name + '_' + line + '.txt',
                    'w')
            fh.write(text + '\n')
            fh.close()
    
    def getReadingOrder(self,element):
        """
        get teh Reading order from xml
        """
        try:
            e_ro =  re.match(r".*readingOrder {.*index:(.*);.*}",
                      element.attrib['custom']).group(1)
        except:
            e_ro = None
            logging.warning('No region readingOrder defined for {} at {}'.format(
                            self.getId(element),
                            self.name))
        return e_ro



    def splitImageByLine(self,img,size):
        """
        Save an PNG image for each line defined on XML-PAGE
        WIP: do not ready for use
        """
        regions = {}
        for i,element in enumerate(self.root.findall(".//"+self.base+'TextRegion')):
            e_ro = self.getReadingOrder(element)
            if (e_ro == None):
                e_ro = i
            regions[e_ro] = element
        
        for ro_region in sorted(regions):
            e_coords = selg.getCoords(regions[ro_region])
            e_id = self.getId(regions[ro_region])
            lines = {}
            for j,line in enumerate(regions[ro_region].findall(".//"+self.base+'TextLine')):
                l_id = self.getId(line)
                l_ro = self.getReadingOrder(line)
                if (l_ro == None):
                    l_ro = j
                lines[l_ro] = (l_id, line)
            prev = e_corrds #--- first element is region boundary 
            for ro_line in sorted(lines):
                ro_line + 1
                





