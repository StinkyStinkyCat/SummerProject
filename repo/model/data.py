import numpy as np
import math
import cv2
from typing import Tuple, Union
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

import settings

# for Mediapipe
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


class FaceRipper:
    def __init__(self, path: str, label='', detection_res=None, standardization=False):
        # t = mp.Image.create_from_file(path)
        # self.img_orig = t.numpy_view()
        self.img_orig = cv2.imread(path)
        self.img_height_orig, self.img_width_orig, channels = np.shape(self.img_orig)
        if channels == 4:
            print('!!!!!!!!!!!!!!!!')
            self.img_orig = self.img_orig[:, :, :3]
        self.img_orig = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2RGB)
        t = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.img_orig)
        self.label = label
        # detect face
        if not detection_res:
            detection_res = detector.detect(t)
        self.detection_res = detection_res
        # calculate original face box
        # up down left right
        self.face_box_orig, self.face_oval_vertices_trace = self.__calculate_orig_face_box()
        # refit the face box to 256 * 256 by resizing the whole image
        self.img_resized, self.resize_ratio, self.face_box_resized = self.__resize_orig_to_input()
        self.img_height_resized, self.img_width_resized, _ = np.shape(self.img_resized)
        # crop the face
        self.shape_mask = self.__draw_shape_mask(self.face_oval_vertices_trace)
        if standardization:
            self.img_resized = tf.image.per_image_standardization(self.img_resized).numpy()
        self.face = self.__crop_face(self.shape_mask)

    # extracted and adapted from the source file from mediapipe of drawing landmarks
    # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py?ref=assemblyai.com
    def __calculate_orig_face_box(self) -> ((int, int, int, int), [(int, int)]):
        """Return a vertices trace of face oval for mask drawing as well"""

        def normalized_to_pixel_coordinates(
                normalized_x: float, normalized_y: float, image_width: int,
                image_height: int) -> Union[None, Tuple[int, int]]:
            """Converts normalized value pair to pixel coordinates."""

            # Checks if the float value is between 0 and 1.
            def is_valid_normalized_value(value: float) -> bool:
                return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
                return None
            x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px

        image = self.img_orig
        face_landmarks = self.detection_res.face_landmarks[0]
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        connections = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        _PRESENCE_THRESHOLD = 0.5
        _VISIBILITY_THRESHOLD = 0.5
        _BGR_CHANNELS = 3

        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                          image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

        # my code
        start_pt = (0, 0)
        pt_num = 0
        line_dict = {}
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                line_dict.update({idx_to_coordinates[start_idx]: idx_to_coordinates[end_idx]})
                pt_num += 1
                start_pt = idx_to_coordinates[start_idx]
        poly_list = [start_pt]
        left_border, upper_border, right_border, lower_border = start_pt[0], start_pt[1], start_pt[0], start_pt[1]
        for i in range(pt_num - 1):
            start_pt = line_dict[start_pt]
            poly_list.append(start_pt)
            if start_pt[1] < upper_border:
                upper_border = start_pt[1]
            if start_pt[1] > lower_border:
                lower_border = start_pt[1]
            if start_pt[0] < left_border:
                left_border = start_pt[0]
            if start_pt[0] > right_border:
                right_border = start_pt[0]
        # can be out of bound
        left_border = left_border - settings.FACE_BOX_MARGIN
        right_border = right_border + settings.FACE_BOX_MARGIN
        upper_border = upper_border - settings.FACE_BOX_MARGIN
        lower_border = lower_border + settings.FACE_BOX_MARGIN
        return (upper_border, lower_border, left_border, right_border), poly_list

    def __resize_orig_to_input(self) -> (np.ndarray, (float, float), (int, int, int, int)):
        """Resize the original input to facebox at 256; Return ratios and the bounding box"""
        up, down, left, right = self.face_box_orig
        face_w_expect, face_h_expect = settings.INPUT_SIZE
        face_w_orig = right - left + 1
        face_h_orig = down - up + 1
        resize_ratio_w = face_w_expect / face_w_orig
        resize_ratio_h = face_h_expect / face_h_orig
        # check the new face box
        right = int(right * resize_ratio_w)
        left = int(left * resize_ratio_w)
        down = int(down * resize_ratio_h)
        up = int(up * resize_ratio_h)
        delta_width = right - left - face_w_expect
        delta_height = down - up - face_h_expect
        # the delta should always <= 2 ?
        if abs(delta_width) > 2 or abs(delta_height) > 2:
            print(int(right * resize_ratio_w) - int(left * resize_ratio_w),
                  int(down * resize_ratio_h) - int(up * resize_ratio_h))
            raise Exception('Failed to scale the face box.')
        if delta_width == -1:
            left -= 1
        if delta_width == -2:
            left -= 1
            right += 1
        if delta_height == -1:
            up -= 1
        if delta_height == -2:
            up -= 1
            down += 1
        img_w_new = int(self.img_width_orig * resize_ratio_w)
        img_h_new = int(self.img_height_orig * resize_ratio_h)
        res = cv2.resize(self.img_orig, (img_w_new, img_h_new), interpolation=settings.RESIZING_INTERPOLATION)
        return res, (resize_ratio_h, resize_ratio_w), (up, down, left, right)

    def __draw_shape_mask(self, face_oval_vertices_trace: [(int, int)]) -> np.ndarray:
        mask = np.zeros(np.shape(self.img_resized), np.uint8)
        # scale the vertices
        for i in range(len(face_oval_vertices_trace)):
            x, y = face_oval_vertices_trace[i]
            face_oval_vertices_trace[i] = (x * self.resize_ratio[1], y * self.resize_ratio[0])
        # ?????????
        face_oval_vertices_trace = np.array(face_oval_vertices_trace)
        face_oval_vertices_trace = np.int32([face_oval_vertices_trace])
        # ?????????
        cv2.fillPoly(mask, pts=face_oval_vertices_trace, color=(255, 255, 255))
        return mask

    def __crop_face(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(self.img_resized, self.img_resized, mask=mask)
        up, down, left, right = self.face_box_resized
        # TODO handle out of bound case
        if left < 0 or right >= self.img_width_resized or up < 0 or down >= self.img_height_resized:
            raise Exception("Out of bound when cropping face")
        face = masked[up:down, left:right]
        return face

    def update_face(self, new_face: np.ndarray) -> np.ndarray:
        """Crop the new face back to the initial image"""
        # must when standardization = True
        res = self.img_resized.copy()
        up, down, left, right = self.face_box_resized
        # res[up:down, left:right] = new_face
        for i in range(0, down - up):
            for j in range(0, right - left):
                b, g, r = self.shape_mask[up + i][left + j]
                if not (b == 0 and g == 0 and r == 0):
                    res[up + i][left + j] = new_face[i][j]
        # de-standardize and resize to original size
        res = tf.keras.utils.array_to_img(res)
        res = np.array(res.convert('RGB'))
        size = (self.img_width_orig, self.img_height_orig)
        res = cv2.resize(res, size, interpolation=settings.RESIZING_INTERPOLATION)
        return res

    def apply_mask_and_update_face(self, color_mask: np.ndarray):
        """Apply a normalized mask to the face and then return the final image"""

        def mask_formula(a, b):
            a = int(a)
            b = int(b)
            ans = a - 255 + 2 * b
            if ans > 255:
                return 255
            elif ans < 0:
                return 0
            return ans

        res = self.img_resized.copy()
        up, down, left, right = self.face_box_resized
        for i in range(0, down - up):
            for j in range(0, right - left):
                r, g, b = self.shape_mask[up + i][left + j]
                if not (b == 0 and g == 0 and r == 0):
                    b1, g1, r1 = res[up + i][left + j]
                    b2, g2, r2 = color_mask[i][j]
                    res[up + i][left + j] = [mask_formula(b1, b2), mask_formula(g1, g2), mask_formula(r1, r2)]
        size = (self.img_width_orig, self.img_height_orig)
        res = cv2.resize(res, size, interpolation=settings.RESIZING_INTERPOLATION)
        return res


def standardize_face(face: np.ndarray):
    """Return zero-centered face, mean and adjusted deviation"""
    mean = np.mean(face)
    ad_dev = max(np.std(face), 1.0 / settings.INPUT_SIZE[0])
    face = (face.astype(np.float64) - mean) / ad_dev
    return face.astype(np.float32), mean, ad_dev


def prepare_mask(mask):
    """Normalize the mask"""
    mask = mask.astype(np.float64)
    mask -= 128
    mask /= 128
    return mask.astype(np.float32)


def de_prepare_mask(mask):
    """De-normalize the mask"""
    mask *= 128
    mask += 128
    return mask.astype(np.uint8)


# def de_standardize_mask(mask):
#     mask = mask - np.min(mask)
#     x_max = np.max(mask)
#     if x_max != 0:
#         mask /= x_max
#     mask *= 255
#     b, g, r = mask[0][0]
#     p = (b + g + r) / 3
#     rate = (255 / 2 - p) / (255 - p)
#     mask = rate * (255 - mask) + mask
#     return mask.astype(np.uint8)


def calculate_color_mask(face1: np.ndarray, face2: np.ndarray) -> np.ndarray:
    """face1 - 255 + 2mask = face2
     Return: an uint8 ndarray
    """
    face1 = face1.astype(np.int16)
    face2 = face2.astype(np.int16)
    res = face2 - face1 + 255
    res = (res / 2).round()
    return res.astype(np.uint8)
