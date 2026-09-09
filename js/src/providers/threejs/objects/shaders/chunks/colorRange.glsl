// Map a scalar onto a colormap. A zero-width range is the map's middle, not Inf/NaN.
// THREE.ShaderChunk.k3d_color_range — raster shaders and the cinematic volume both include this.
float k3dScaleToRange(float value, float lo, float hi) {
    return hi != lo ? (value - lo) / (hi - lo) : 0.5;
}
