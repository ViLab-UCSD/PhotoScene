import bpy
import sys

# blender -b -P photoscene/applyUV.py -- inOBJ outOBJ

# Parse args if in bg mode.
if bpy.app.background:
    args = sys.argv[sys.argv.index("--") + 1:]
    mesh_file = args[0]
    mesh_file_out = args[1]
else:
    print('Please use -b to use background mode')
    assert(False)

# Delete objects.
for obj in bpy.context.scene.objects:
    obj.select_set(True)

bpy.ops.object.delete()

# Add mesh.
bpy.ops.import_scene.obj(filepath=str(mesh_file))

obj = bpy.context.selected_objects[0]
# Select each object
obj.select_set(True)
# Make it active
bpy.context.view_layer.objects.active = obj
# Toggle into Edit Mode
bpy.ops.object.mode_set(mode='EDIT')
# Select the geometry
bpy.ops.mesh.select_all(action='SELECT')
# Call the smart project operator
bpy.ops.uv.smart_project()
# Toggle out of Edit Mode
bpy.ops.object.mode_set(mode='OBJECT')
# Deselect the object
obj.select_set(False)

bpy.ops.export_scene.obj(filepath=mesh_file_out, use_materials=True)