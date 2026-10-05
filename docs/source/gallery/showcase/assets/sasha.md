# sasha.glb

"Sasha" by saber7711, https://blendswap.com/blend/29574, licensed CC-BY
(https://creativecommons.org/licenses/by/4.0/).

Changed from the original `Sasha.blend`: exported to glTF with Blender 5.2 by `sasha_from_blend.py`,
which rebuilds the Cycles materials as Principled BSDF (glossy metals as metals, the three added
glass shaders of the stones as one transmissive material with IOR 2.09 and a thickness, so
glTF reads them as solids rather than thin walls), keeps the ring only - no
camera, studio light planes or floor - applies the modifiers at their render levels, and
compresses the meshes with Draco.

    blender -b Sasha.blend --python sasha_from_blend.py -- sasha.glb
