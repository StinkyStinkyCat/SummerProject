import cv2

file_format = '.png'

original_name = '_orig'
albedo_name = '_albedo'
butterfly_name = '_butt'
rembrandt_name = '_remb'
split_name = '_split'

name_list = (original_name, albedo_name, butterfly_name, rembrandt_name, split_name)

INPUT_SIZE = (256, 256)
RESIZING_INTERPOLATION = cv2.INTER_LINEAR
FACE_BOX_MARGIN = 10
