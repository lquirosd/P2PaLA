from __future__ import print_function
from __future__ import division

import os
import logging

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import re
import datetime


class pageData():
    """ Class to process PAGE xml files"""
    def __init__(self,filepath,logger=None,creator=None):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.logger = logging.getLogger(__name__) if logger==None else logger
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.creator = 'P2PaLA-PRHLT' if creator==None else creator
        self.XMLNS= {'xmlns':"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                      'xmlns:xsi':"http://www.w3.org/2001/XMLSchema-instance", 
                      'xsi:schemaLocation':"""http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 
                      http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"""}
        #self.parse()


    def parse(self):
        """
        Parse PAGE-XML file
        """
        tree = ET.parse(self.filepath)
        #--- get the root of the data
        self.root = tree.getroot()
        #--- save "namespace" base
        self.base = self.root.tag.rsplit('}',1)[0] + '}'
    
    def get_region(self,region_name):
        """
        get all regions in PAGE which match region_name
        """
        return self.root.findall(".//"+self.base+region_name) or None

    def get_id(self,element):
        """
        get Id of current element
        """
        return str(element.attrib.get('id'))

    
    def get_region_type(self, element):
        """
        Returns the type of element
        """
        try:
            e_type =  re.match(r".*structure {.*type:(.*);.*}",
                      element.attrib['custom']).group(1)
        except:
            e_type = None
            self.logger.warning('No region type defined for {} at {}'.format(
                            self.get_id(element),
                            self.name))
        return e_type

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        img_width = int(self.root.findall("./"+self.base+"Page")[0].get(
                        'imageWidth'))
        img_height = int(self.root.findall("./"+self.base+"Page")[0].get(
                        'imageHeight'))
        return (img_width, img_height)

    def get_coords(self, element):
        str_coords = element.findall('./' + self.base +
                                    'Coords')[0].attrib.get('points').split()
        return np.array([i.split(',') for i in str_coords]).astype(np.int)


    def build_mask(self,out_size,element_name, color_dic):
        """
        Builds a "image" mask of desired elements
        """
        size = self.get_size()[::-1]
        mask = np.zeros(out_size, np.uint8)
        scale_factor = out_size/size
        #mask = np.zeros((int(size[1]*scale_factor),
        #                 int(size[0]*scale_factor)), np.uint8) + 255
        for element in self.root.findall(".//"+self.base+element_name):
            #--- get element type
            e_type = self.get_region_type(element)
            if (e_type == None or e_type not in color_dic):
                e_color = 175
                self.logger.warning('Element type "{}"undefined on color dic, set to default=175'.format(e_type))
                print('Element type "{}"undefined on color dic, set to default=175'.format(e_type))

            else:
                e_color = color_dic[e_type]
            #--- get element coords
            coords = self.get_coords(element)
            coords = (coords * np.flip(scale_factor,0)).astype(np.int)
            cv2.fillConvexPoly(mask,coords,e_color)
        return mask

    def build_baseline_mask(self,out_size,color,line_width):
        """
        Builds a "image" mask of Baselines on XML-PAGE
        """
        size = self.get_size()[::-1]
        mask = np.zeros((out_size[0],out_size[1]), np.uint8)
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
    
    def get_text(self,element):
        """
        get Text defined for element
        """
        text_node = element.find('./' + self.base + 'TextEquiv')
        if (text_node == None):
            self.logger.warning('No Text node found for line {} at {}'.format(
                            self.get_id(element),
                            self.name))
            return ''
        else:
            text_data = text_node.find('*').text
            if (text_data == None):
                self.logger.warning('No text found in line {} at {}'.format(
                                self.get_id(element),
                                self.name))
                return ''
            else:
                return text_data.encode('utf-8').strip()

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.root.findall(".//"+self.base+'TextRegion'):
            r_id = self.get_id(element)
            for line in element.findall(".//"+self.base+'TextLine'):
                l_id = self.get_id(line)
                data[r_id + '_' + l_id] = self.get_text(line)

        return data

    def write_transcriptions(self,out_dir):
        """write out one txt file per text line"""
        for line,text in self.get_transcription().iteritems():
            fh = open (out_dir + '/' + self.name + '_' + line + '.txt',
                    'w')
            fh.write(text + '\n')
            fh.close()
    
    def get_reading_order(self,element):
        """get the Reading order of `element` from xml data"""
        try:
            e_ro =  re.match(r".*readingOrder {.*index:(.*);.*}",
                      element.attrib['custom']).group(1)
        except:
            e_ro = None
            self.logger.warning('No region readingOrder defined for {} at {}'.format(
                            self.get_id(element),
                            self.name))
        return e_ro

    def split_image_by_line(self,img,size):
        """save an PNG image for each line defined on XML-PAGE"""
        raise NotImplementedError
        regions = {}
        for i,element in enumerate(self.root.findall(".//"+self.base+'TextRegion')):
            e_ro = self.get_reading_order(element)
            if (e_ro == None):
                e_ro = i
            regions[e_ro] = element
        
        for ro_region in sorted(regions):
            e_coords = selg.get_coords(regions[ro_region])
            e_id = self.get_id(regions[ro_region])
            lines = {}
            for j,line in enumerate(regions[ro_region].findall(".//"+self.base+'TextLine')):
                l_id = self.get_id(line)
                l_ro = self.get_reading_order(line)
                if (l_ro == None):
                    l_ro = j
                lines[l_ro] = (l_id, line)
            prev = e_corrds #--- first element is region boundary 
            for ro_line in sorted(lines):
                ro_line + 1
    
    def new_page(self,name, rows,cols):
        """create a new PAGE xml"""
        self.xml = ET.Element('PcGts')
        self.xml.attrib = self.XMLNS
        metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(metadata, 'Creator').text = self.creator
        ET.SubElement(metadata, 'Created').text = datetime.datetime.today().strftime('%Y-%m-%dT%X')
        ET.SubElement(metadata, 'LastChange').text = datetime.datetime.today().strftime('%Y-%m-%dT%X')
        self.page = ET.SubElement(self.xml, 'Page')
        self.page.attrib = {'imageFilename':name,
                            'imageWidth':rows,
                            'imageHeight':cols}
    
    def add_element(self, r_class, r_id, r_type, r_coords, parent=None):
        """add element to parent node"""
        parent = self.page if parent==None else parent
        t_reg = ET.SubElement(parent, r_class)
        t_reg.attrib = {'id':r_class + '_' + str(r_id),
                        'custom':'structure {type:' + r_type + ';}'}
        ET.SubElement(t_reg,'Coords').attrib = {'points':r_coords}
        return t_reg

    def remove_element(self, element, parent=None):
        """remove element from parent node"""
        parent = self.page if parent==None else parent
        parent.remove(element)


    def add_baseline(self, b_coords, parent):
        """add baseline element ot parent line node"""
        ET.SubElement(parent,'Baseline').attrib = {'points':b_coords}

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        tree.write(self.filepath,encoding="UTF-8", xml_declaration=True)

    def _indent(self,elem, level=0):
        """
        Function borrowed from: 
            http://effbot.org/zone/element-lib.htm#prettyprint
        """
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
                

