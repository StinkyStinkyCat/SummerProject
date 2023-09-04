import pyautogui
import time
import os

import client

# interval time between each clicking or typing
sleep_time = 0.1
# prepare time for setting up the windows
prepare_time = 3

# path for face gen output
# C:\Users\[USER_NAME]\Documents\DAZ 3D\Studio\My Library\data\DAZ 3D\Genesis 8\[GENDER] 8_1\Morphs\FaceGen
facegen_output_path_m = os.path.join(
    os.path.expanduser('~'),
    'Documents', 'DAZ 3D', 'Studio', 'My Library',
    'data', 'DAZ 3D', 'Genesis 8')
facegen_output_path_f = os.path.join(facegen_output_path_m, 'Female 8_1', 'Morphs', 'FaceGen', 'facegen.dsf')
facegen_output_path_m = os.path.join(facegen_output_path_m, 'Male 8_1', 'Morphs', 'FaceGen', 'facegen.dsf')

daz_preset_path_f = os.path.join(
    os.path.expanduser('~'),
    'Documents', 'DAZ 3D', 'Studio', 'My Library',
    'Presets', 'Characters')
daz_preset_path_m = os.path.join(daz_preset_path_f, 'facegen_m.duf')
daz_preset_path_f = os.path.join(daz_preset_path_f, 'facegen_f.duf')

# =============== FACEGEN ===============
# ===== pos for generating =====
pos_tab_create = (1135, 33)
pos_tab_new = (1194, 63)
pos_op_male = (1300, 146)
pos_op_female = (1300, 173)
pos_but_generate = (1832, 534)
# =====  pos for modifying =====
pos_tab_modify = (1188, 32)
pos_tab_color = (1304, 61)
pos_slider_thick_brow = (2444, 737)
# ===== pos for outputting =====
pos_tab_file = (1359, 32)
pos_tab_export = (1194, 63)
pos_box_file_name = (1333, 169)
pos_tab_genesis_81 = (1474, 200)
pos_tab_female = (1164, 222)
pos_tab_male = (1212, 226)
pos_but_restart = (1690, 259)
pos_but_export = (1723, 955)
# =============== DAZ3D ===============
# ===== pos setup actor =====
pos_actor = (1333, 588)
pos_facegen_f = (175, 208)
pos_facegen_m = (293, 212)
pos_file = (13, 30)
pos_send_to = (103, 289)
pos_daz_to_blender = (262, 366)
# =============== Blender ===============
pos_import_figure = (2004, 149)
pos_remove_all_daz = (2001, 472)
# ===== Windows Tab Position
pos_facegen_window = (1374, 1418)
pos_daz_window = (1431, 1419)
pos_blender_window = (1468, 1418)


def type_string(s: str):
    pyautogui.write(s)
    time.sleep(sleep_time)


def press_key(s: str):
    pyautogui.press(s)
    time.sleep(sleep_time)


def left_click(pos: tuple):
    pyautogui.click(pos[0], pos[1])
    time.sleep(sleep_time)


def alt_tab(n: int):
    pyautogui.keyDown('alt')
    for i in range(n):
        pyautogui.press('tab')
    pyautogui.keyUp('alt')
    time.sleep(sleep_time)


def macro_facegen(gender: str):
    # navigate to create main tab
    left_click(pos_tab_create)
    # navigate to new sub tab
    left_click(pos_tab_new)
    # set gender
    if gender == 'm':
        left_click(pos_op_male)
    else:
        left_click(pos_op_female)
    # generate face now
    left_click(pos_but_generate)
    # navigate to modify tab
    left_click(pos_tab_modify)
    # navigate to color sub tab
    left_click(pos_tab_color)
    # make brow thicker
    left_click(pos_slider_thick_brow)
    left_click(pos_slider_thick_brow)
    left_click(pos_slider_thick_brow)
    # navigate to file main tab
    left_click(pos_tab_file)
    # navigate to export sub tab
    left_click(pos_tab_export)
    # change file name to "facegen"
    left_click(pos_box_file_name)
    left_click(pos_box_file_name)
    type_string("facegen")
    # navigate to genesis 8.1 sub sub tab
    left_click(pos_tab_genesis_81)
    # navigate to gender sub sub sub tab
    if gender == 'm':
        left_click(pos_tab_male)
    else:
        left_click(pos_tab_female)
    # click restart button (will appear at second time)
    left_click(pos_but_restart)
    # finally export to daz
    left_click(pos_but_export)


def macro_facegen_to_daz(gender: str):
    # delete the former daz face file
    if gender == 'm' and os.path.exists(facegen_output_path_m):
        os.remove(facegen_output_path_m)
    elif os.path.exists(facegen_output_path_f):
        os.remove(facegen_output_path_f)
    # generate a face
    macro_facegen(gender)
    # wait for exporting to complete
    time.sleep(6)
    if gender == 'm':
        while not os.path.isfile(os.path.join(facegen_output_path_m)):
            time.sleep(1)
    else:
        while not os.path.isfile(os.path.join(facegen_output_path_f)):
            time.sleep(1)
    time.sleep(0.5)


def add_expression():
    pass


def remove_expression():
    pass


def change_hair(gender: str):
    pass


def macro_daz_to_blender(gender: str):
    # add figure
    if gender == 'f':
        left_click(pos_facegen_f)
        left_click(pos_facegen_f)
    else:
        left_click(pos_facegen_m)
        left_click(pos_facegen_m)
    time.sleep(5)
    # import to blender in menu
    left_click(pos_file)
    left_click(pos_send_to)
    left_click(pos_daz_to_blender)
    # press enter to start
    press_key('enter')
    time.sleep(7)
    # press enter to clear the finishing notice
    press_key('enter')
    # delete the figure
    left_click(pos_actor)
    press_key('delete')


def macro_full(gender: str):
    macro_facegen_to_daz(gender)
    left_click(pos_daz_window)
    change_hair(gender)
    macro_daz_to_blender(gender)
    left_click(pos_facegen_window)


def generate(num_m: int, num_f: int, c: client.Client):
    """Start the whole process, activate Blender sever first !!!"""
    for i in range(num_m):
        macro_full('m')
        if i != 0:
            c.wait_for_finished_message()
        c.send_render_request()
    for i in range(num_f):
        macro_full('f')
        if not (num_m == 0 and i == 0):
            c.wait_for_finished_message()
        c.send_render_request()
    c.wait_for_finished_message()
    c.send_stop_request()


if __name__ == "__main__":
    time.sleep(prepare_time)
    left_click(pos_facegen_window)
    generate(num_m=10, num_f=10, c=client.Client())
