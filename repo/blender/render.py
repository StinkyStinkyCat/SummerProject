import bpy
import os
import random
import math
import time


class BlenderController:
    camera_distance = 1.25
    camera_height = 1.67
    count = 0

    light_set_remb = ('Rembrandt',)
    light_set_split = ('Split', 'Split - Supp')
    light_set_butt = ('Butterfly',)

    def __init__(self,
                 dir_folder=os.path.join(os.path.expanduser("~"), "Output"),
                 sun_elevation_range=(15, 90),
                 camera_angle_range_horizontal=(-20, 20),
                 camera_angle_range_vertical=(-5, 5),
                 camera_angle_range_tilt=(-5, 5)):
        self.dir_folder = dir_folder
        self.sun_elevation_range = sun_elevation_range
        self.camera_angle_range_horizontal = camera_angle_range_horizontal
        self.camera_angle_range_vertical = camera_angle_range_vertical
        self.camera_angle_range_tilt = camera_angle_range_tilt

    def __switch_light_set(self, light_set_names: (str), b: bool):
        for name in light_set_names:
            bpy.context.view_layer.objects[name].data.node_tree.nodes['Emission'].inputs[
            'Strength'].default_value = b * 1

    def __switch_background_light(self, b: bool):
        bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = b * 1

    def __render(self, file_name: str):
        bpy.context.scene.render.filepath = os.path.join(self.dir_folder, file_name)
        bpy.ops.render.render(write_still=True)

    def randomize_sun(self):
        sky_node = bpy.context.scene.world.node_tree.nodes['Sky Texture']
        sky_node.sun_intensity = random.uniform(0.67, 1.75)
        sky_node.sun_elevation = math.radians(
            random.randint(*self.sun_elevation_range)
        )
        sky_node.sun_rotation = math.radians(random.randint(0, 359))

    def randomize_camera(self):
        camera = bpy.context.view_layer.objects['Camera']
        r = self.camera_distance
        alpha = random.randint(*self.camera_angle_range_horizontal)
        y = - r * math.cos(math.radians(alpha))
        x = r * math.sin(math.radians(alpha))
        z = camera.location[2]
        beta = random.randint(*self.camera_angle_range_vertical)
        x = x * math.cos(math.radians(beta))
        y = y * math.cos(math.radians(beta))
        z = self.camera_height + r * math.sin(math.radians(beta))
        gama = random.randint(*self.camera_angle_range_tilt)
        camera.location = (x, y, z)
        camera.rotation_euler = (math.pi / 2 - math.radians(beta), math.radians(gama), math.radians(alpha))

    def reset_camera(self):
        camera = bpy.context.view_layer.objects['Camera']
        camera.location = (0, -self.camera_distance, self.camera_height)
        camera.rotation_euler = (math.pi / 2, 0, 0)

    def __switch_output_pass(self, keyword: str):
        node_tree = bpy.context.scene.node_tree
        render_layer_node = node_tree.nodes['Render Layers']
        composite_node = node_tree.nodes['Composite']
        if keyword == 'image':
            node_tree.links.new(render_layer_node.outputs['Image'], composite_node.inputs['Image'])
        if keyword == 'diffcol':
            node_tree.links.new(render_layer_node.outputs['DiffCol'], composite_node.inputs['Image'])

    def render_all(self, file_name: str):
        # import the figure
        bpy.ops.import_.fbx()
        # randomize camera and sun
        self.randomize_camera()
        self.randomize_sun()
        # render general version
        self.__switch_output_pass('image')
        self.__switch_light_set(self.light_set_butt, False)
        self.__switch_light_set(self.light_set_remb, False)
        self.__switch_light_set(self.light_set_split, False)
        self.__switch_background_light(True)
        self.__render(file_name + '_orig.png')
        # no background from now on
        self.__switch_background_light(False)
        # render lighted version 1
        self.__switch_light_set(self.light_set_butt, True)
        self.__render(file_name + '_butt.png')
        self.__switch_light_set(self.light_set_butt, False)
        # render lighted version 2
        self.__switch_light_set(self.light_set_remb, True)
        self.__render(file_name + '_remb.png')
        self.__switch_light_set(self.light_set_remb, False)
        # render lighted version 3
        self.__switch_light_set(self.light_set_split, True)
        self.__render(file_name + '_split.png')
        self.__switch_light_set(self.light_set_split, False)
        # render albedo version
        self.__switch_output_pass('diffcol')
        time.sleep(0.5)
        self.__render(file_name + '_albedo.png')
        # remove the figure
        bpy.ops.remove.alldaz()
