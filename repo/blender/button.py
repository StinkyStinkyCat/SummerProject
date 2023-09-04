import sys
sys.path.append('C://Users//18501//Documents//BlenderScripts')

# =============== IMPORTANT ===============
# Need to modify the Daz to Blender addon source file
# Rename the bl_idname "import.fbx" lebel in DtbPanels.py and DtbOperators.py to something like "import_.fbx"
# Or you will have to call "bpy.ops.import.fbx()" which will be rejected by python because of "import"

import bpy
import render
import server

class PuppyStartOperator(bpy.types.Operator):
    """Start the Puppy Server!!!"""
    bl_idname = "puppy.init"
    bl_label = "puppy.init"

    def execute(self, context):
        ppserver = server.PuppyServer()
        ppserver.handle()
        return {"FINISHED"}


class PuppyTestOperator(bpy.types.Operator):
    """Test the script!!!"""
    bl_idname = "puppy.test"
    bl_label = "puppy.test"

    def execute(self, context):
        t = render.BlenderController()
        t.randomize_camera()
        # t.randomize_sun()
        # t.render_all('test')
        # t.reset_camera()
        return {"FINISHED"}


class VIEW3D_PT_puppy_panel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Puppy"
    bl_label = "Puppy"

    def draw(self, context):
        """Puppy tools"""
        row = self.layout.row()
        row.operator("puppy.init" ,text="Start!!!")
        row = self.layout.row()
        row.operator("puppy.test" ,text="Test!!!")

def register():
    bpy.utils.register_class(VIEW3D_PT_puppy_panel)
    bpy.utils.register_class(PuppyStartOperator)
    bpy.utils.register_class(PuppyTestOperator)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_puppy_panel)
    bpy.utils.unregister_class(PuppyStartOperator)
    bpy.utils.unregister_class(PuppyTestOperator)

if __name__ == "__main__":
    register()
