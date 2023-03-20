import xml.etree.ElementTree as ET
from xml.dom import minidom


class XmlHandler:
    def __init__(self):
        pass

    def create_element(self, name, **kwargs):
        element = ET.Element(name)
        if kwargs.get('comment'):
            comment = kwargs.get('comment')
            element.append(comment)
        if kwargs.get('text'):
            element.text = kwargs.get('text')
        if kwargs.get('tail'):
            element.tail = kwargs.get('tail')
        if kwargs.get('attrib'):
            element.attrib = kwargs.get('attrib')
        return element

    def add_childs(self, root, childs=[]):
        for child in childs:
            root.append(child)

    def print_xml(self, xml, pretty=True):
        if pretty:
            print(self.prettify(xml))
        else:
            print(ET.tostring(xml, 'unicode', method='xml'))

    def save_xml(self, xml, path, pretty=True):
        with open(path, 'w') as f:
            if pretty:
                f.write(self.prettify(xml))
            else:
                f.write(ET.tostring(xml, 'unicode', method='xml'))

    def prettify(self, xml):
        rough_string = ET.tostring(xml, 'unicode', method='xml')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent='  ')

    def parse(self, xml_path):
        return ET.parse(xml_path)

    def get_root(self, xml):
        return xml.getroot()
