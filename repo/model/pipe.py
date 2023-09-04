import cv2
import os
import numpy as np
import random

import settings
import data


class Pipe:
    def __init__(self, path: str):
        self.path = path
        self.index_max = len(os.listdir(self.path)) // 2
        self.input_index_list = list(range(1, self.index_max + 1))
        random.shuffle(self.input_index_list)

    # def load_next(self) -> InputDataPair | None:
    #     if len(self.input_index_list) == 0:
    #         return None
    #     serial = self.input_index_list.pop()
    #     print('Loading index: ' + str(serial))
    #     file_a, file_b = self._calculate_file_path(serial)
    #     return InputDataPair(file_a, '1', file_b, '2')

    def _calculate_file_path(self, serial: int):
        name = str(serial).zfill(3)
        return os.path.join(self.path, name + settings.postfix_original + settings.file_format), \
            os.path.join(self.path, name + settings.postfix_albedo + settings.file_format)


def build_face_cache():
    """Crop all the faces and save"""
    index_max = len(os.listdir('input')) // len(settings.name_list)
    for i in range(1, index_max + 1):
        name = str(i).zfill(3)
        # data_set = InputDataSet()
        # for j in range(len(settings.name_list)):
        #     path = 'input/' + name + settings.name_list[j] + settings.file_format
        #     data_set.load_data(path)
        # for j in range(len(settings.name_list)):
        #     path = 'input_face_cache/' + name + settings.name_list[j] + settings.file_format
        #     # cv2.imwrite(path, data_set.get_face(j))
        #     cv2.imwrite(path, data_set.data_list[j].img_orig)
        face_rippers = [data.FaceRipper('input/' + name + settings.name_list[0] + settings.file_format)]
        for j in range(1, len(settings.name_list)):
            face_rippers.append(data.FaceRipper('input/' + name + settings.name_list[j] + settings.file_format,
                                                detection_res=face_rippers[0].detection_res))
        for j in range(len(settings.name_list)):
            path = 'input_face_cache/' + name + settings.name_list[j] + settings.file_format
            cv2.imwrite(path, face_rippers[j].face)


def build_color_mask_cache():
    """Build the color masks"""
    index_max = len(os.listdir('input')) // len(settings.name_list)
    for i in range(1, index_max + 1):
        name = str(i).zfill(3)
        faces = []
        for j in range(0, len(settings.name_list)):
            faces.append(cv2.imread('input_face_cache/' + name + settings.name_list[j] + settings.file_format))
        for j in range(2, len(settings.name_list)):
            path = 'color_mask_cache/' + name + settings.name_list[j] + settings.file_format
            mask = data.calculate_color_mask(faces[0], faces[j])
            cv2.imwrite(path, mask)


if __name__ == '__main__':
    build_face_cache()
    build_color_mask_cache()
