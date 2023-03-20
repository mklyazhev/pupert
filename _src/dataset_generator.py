import os
import numpy as np
import cv2
import re
import shutil
from math import ceil
from xml_handler import XmlHandler
from eye_detector import EyesDetector
from sota import PupilsDetectorSOTA
import utils


class DatasetGenerator:
    def __init__(self, dir_from, dir_to, n_samples=None, skip_pattern='^.*_([2-9]|\d{2}).*$',
                 target_extensions=('.png', '.jpg'), delete_origin=False, verbose=False):
        self.dir_from = dir_from
        self.dir_to = dir_to
        self.skip_pattern = skip_pattern
        self.target_extensions = target_extensions
        self.delete_origin = delete_origin
        self.verbose = verbose
        self.n_samples = n_samples if n_samples else self.__scan_dir()
        self.eyes_detector = EyesDetector()
        self.pupils_detector = PupilsDetectorSOTA()
        self.xml_handler = XmlHandler()
        self.samples_xml = self.xml_handler.create_element('images')

    def __scan_dir(self):
        count = 0
        for root, dirs, files in os.walk(self.dir_from, topdown=True):
            count += len([file for file in files if file.endswith(self.target_extensions) and not self.__check_skip_pattern(file)])
        return count

    def gen_dataset(self):
        if self.verbose:
            print(f'[INFO] Count samples = {self.n_samples}')
        self.__process_dir()
        dataset_xml = self.__create_label_xml(self.samples_xml)
        self.xml_handler.save_xml(dataset_xml, os.path.join(dir_to, 'ibug_300W_pupils.xml'))
        if self.verbose:
            print('\n[INFO] Dataset generated successfully')
        
    def __process_dir(self):
        count_processed_samples = 0
        for root, dirs, files in os.walk(self.dir_from, topdown=True):
            for file in files:
                if file.endswith(self.target_extensions) and not self.__check_skip_pattern(file):
                    self.__process_img(os.path.join(root, file))
                    count_processed_samples += 1
                    if self.verbose:
                        print(
                            f'\r[INFO] Processed: {count_processed_samples}/{self.n_samples}',
                            end='',
                            flush=True
                        )
                    if count_processed_samples == self.n_samples:
                        return

    def __check_skip_pattern(self, img_name):
        if re.search(self.skip_pattern, img_name):
            return True
        else:
            return False

    def __process_img(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
            pupil_coords = self.pupils_detector.predict(img)
            eye_rects = self.eyes_detector.predict(img)
            for i, eye in enumerate(eye_rects):
                pupil_coord = pupil_coords[i]
                if not utils.is_pupil_in_eye_box(pupil_coord, eye):
                    raise Exception('Detected pupil out of eye box')
                if i == 0:
                    side = '_l'
                else:
                    side = '_r'
                img_name = self.__get_name(img_path, side, '.jpg')
                self.__add_img(img_path, img_name)
                self.__append_xml(img, img_name, eye, pupil_coord)
            if self.delete_origin:
                self.__delete_origin(img_path)
        except Exception as e:
            if self.verbose:
                print(f'\n[ERROR] There was an error processing sample -- {os.path.basename(img_path)}: {e}')
            return

    def __add_img(self, img_path, img_name):
        new_path = os.path.join(self.dir_to, img_name)
        shutil.copy2(img_path, new_path)

    def __append_xml(self, img, img_name, eye_rect, pupil_coord):
        pupil_attrib = {'name': '00', 'x': str(pupil_coord[0]), 'y': str(pupil_coord[1])}
        box_attrib = {'top': str(eye_rect[1]), 'left': str(eye_rect[0]), 'width': str(eye_rect[2] - eye_rect[0]), 'height': str(eye_rect[3] - eye_rect[1])}
        image_attrib = {'file': str(img_name), 'width': str(img.shape[1]), 'height': str(img.shape[0])}
        pupil = self.xml_handler.create_element('part', **{'attrib': pupil_attrib})
        box = self.xml_handler.create_element('box', **{'attrib': box_attrib})
        image = self.xml_handler.create_element('image', **{'attrib': image_attrib})
        self.xml_handler.add_childs(box, [pupil])
        self.xml_handler.add_childs(image, [box])
        self.xml_handler.add_childs(self.samples_xml, [image])

    def __get_name(self, img_path, add, extension):
        cur_name = os.path.basename(img_path)
        name, ext = os.path.splitext(cur_name)
        name += add
        new_name = f'{name}{extension}'
        return new_name

    def __delete_origin(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def __create_label_xml(self, samples_xml):
        name_text = 'iBUG pupils coords dataset - All images.'
        comment_text = \
            '''This folder contains data downloaded from:
  http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
  The dataset is actually a combination of the AFW, HELEN, iBUG, and LFPW
  face images.'''

        # prepare main xml nodes
        dataset = self.xml_handler.create_element('dataset')
        name = self.xml_handler.create_element('name', **{'text': name_text})
        comment = self.xml_handler.create_element('comment', **{'text': comment_text})
        images = self.xml_handler.create_element('images')

        # add samples_xml to images node as child
        self.xml_handler.add_childs(images, samples_xml)

        # add childs to main xml
        childs = [name, comment, images]
        self.xml_handler.add_childs(dataset, childs)

        return dataset

    def train_test_split(self, xml_path, split_fraction=None, n_test_samples=None):
        train = self.xml_handler.create_element('images')
        test = self.xml_handler.create_element('images')
        samples_xml = self.xml_handler.parse(xml_path)
        root = self.xml_handler.get_root(samples_xml)
        train_idx, test_idx = self.__get_train_test_idx(root.findall('images/'), split_fraction, n_test_samples)
        for (i, image_xml) in enumerate(root.findall('images/')):
            if i in test_idx:
                self.xml_handler.add_childs(test, [image_xml])
            else:
                self.xml_handler.add_childs(train, [image_xml])
        train_xml = self.__create_label_xml(train)
        test_xml = self.__create_label_xml(test)
        train_xml_name = self.__get_name(xml_path, '_train', '.xml')
        test_xml_name = self.__get_name(xml_path, '_test', '.xml')
        self.xml_handler.save_xml(train_xml, os.path.join(dir_to, train_xml_name))
        self.xml_handler.save_xml(test_xml, os.path.join(dir_to, test_xml_name))

    def __get_train_test_idx(self, root_childs, split_fraction=0.7, n_test_samples=None):
        n_samples = len(root_childs)
        if n_test_samples:
            split_fraction = round(n_test_samples / n_samples, 2)
        else:
            split_fraction = split_fraction
        train_count = ceil(n_samples * split_fraction)
        idx_range = np.arange(0, n_samples)
        rng = np.random.default_rng()
        train_idx = rng.choice(idx_range, size=train_count, replace=False)
        test_idx = idx_range[~np.isin(idx_range, train_idx)]
        return train_idx, test_idx

    def gen_short_dataset(self, xml_path, n_short_samples=None):
        short = self.xml_handler.create_element('images')
        samples_xml = self.xml_handler.parse(xml_path)
        root = self.xml_handler.get_root(samples_xml)
        short_idx = self.__get_short_idx(root.findall('images/'), n_short_samples)
        for (i, image_xml) in enumerate(root.findall('images/')):
            if i in short_idx:
                self.xml_handler.add_childs(short, [image_xml])
        short_xml = self.__create_label_xml(short)
        short_xml_name = self.__get_name(xml_path, '_short', '.xml')
        self.xml_handler.save_xml(short_xml, os.path.join(dir_to, short_xml_name))

    def __get_short_idx(self, root_childs, n_short_samples=None):
        n_samples = len(root_childs)
        idx_range = np.arange(0, n_samples)
        rng = np.random.default_rng()
        short_idx = rng.choice(idx_range, size=n_short_samples, replace=False)
        return short_idx


if __name__ == '__main__':
    dir_from = '../../data/ibug_300W_large_face_landmark_dataset'
    dir_to = '../data'
    handler = DatasetGenerator(dir_from, dir_to, verbose=True)
    handler.gen_dataset()
