"""Sasha.blend (saber7711, CC-BY) to glTF: Cycles node materials rebuilt as Principled BSDF."""
import sys

import bpy
from io_scene_gltf2.blender.com.material_helpers import create_settings_group

OUT = sys.argv[sys.argv.index("--") + 1]

# Cycles materials of the scene, as the glTF exporter understands them (colours are linear)
PRINCIPLED = {
    # white metal of the prongs and settings: Glossy, GGX
    "Material.001": dict(color=(0.8, 0.8, 0.8), metallic=1.0, roughness=0.15),
    # the band: a layer-weight mix of two glossy golds, averaged
    "Material.002": dict(color=(0.5729, 0.3835, 0.1187), metallic=1.0, roughness=0.05),
    # the hallmark plate
    "Material.005": dict(color=(0.07, 0.07, 0.07), metallic=1.0, roughness=0.05),
    # the stones: three added glass BSDFs at IOR 2.08 / 2.09 / 2.095 - dispersion - as one
    "Material.007": dict(color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.05, transmission=1.0, ior=2.09),
}

for name, params in PRINCIPLED.items():
    mat = bpy.data.materials[name]
    tree = mat.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*params["color"], 1.0)
    bsdf.inputs["Metallic"].default_value = params["metallic"]
    bsdf.inputs["Roughness"].default_value = params["roughness"]
    if "transmission" in params:
        bsdf.inputs["Transmission Weight"].default_value = params["transmission"]
        bsdf.inputs["IOR"].default_value = params["ior"]
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

# glass in Cycles is always a solid; glTF needs a thickness for that (KHR_materials_volume),
# else it reads a transmissive surface as a thin wall that light crosses unbent
# the add-on's own group: with its Occlusion socket the exporter keeps the node tree instead of inlining it
group = create_settings_group("glTF Material Output")
stones = bpy.data.materials["Material.007"].node_tree
settings = stones.nodes.new("ShaderNodeGroup")
settings.node_tree = group
settings.inputs["Thickness"].default_value = 1.0

# subdivided as the author rendered it: the viewport level is one higher on the prongs
for obj in bpy.data.objects:
    for modifier in obj.modifiers:
        if modifier.type == "SUBSURF":
            modifier.levels = modifier.render_levels

# the ring only: no camera, studio lights (emissive planes), floor or helpers
keep = [obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name not in ("Plane.017",) and not obj.name.startswith("Свет")]
bpy.ops.object.select_all(action="DESELECT")
for obj in keep:
    obj.select_set(True)
print("EXPORTING", len(keep), "objects")

bpy.ops.export_scene.gltf(
    filepath=OUT,
    export_format="GLB",
    use_selection=True,
    export_apply=True,
    export_yup=True,
    export_cameras=False,
    export_lights=False,
    export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=7,
)
print("WROTE", OUT)
