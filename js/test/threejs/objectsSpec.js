require(['k3d'], function (lib) {
    var K3D = lib.K3D;
    var ThreeJsProvider = lib.ThreeJsProvider;
    var jsonLoader = TestHelpers.jsonLoader;

    const RESAMBLEThreshold = 0.06;

    describe('Objects tests', function () {
        'use strict';

        beforeAll(function (done) {
            WebFont.load({
                custom: {
                    families: ['Lato']
                },
                active: done
            });
        });

        beforeEach(function () {
            this.canvas = TestHelpers.createTestCanvas();
            window.K3DInstance = this.K3D = K3D(ThreeJsProvider, this.canvas, {antialias: 3, axesHelper: false});
        });

        afterEach(function () {
            this.K3D.disable();
            this.K3D = null;
            TestHelpers.removeTestCanvas(this.canvas);
        });

        it('2 torus knots should be drawn as on reference image', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/torus-knots.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'torus-knots', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model without colors should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_without_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_without_colors', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and opacity should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_opacity.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_opacity', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=3d should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_3d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_3d',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=dot should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_dot.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_dot',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=flat should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_flat.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_flat',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=mesh should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_mesh.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_mesh',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple single line text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_text_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'simple_text_label', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple multiline text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_multiline_text_label.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'simple_multiline_text_label',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a text with fill and stroke colors', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/text_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'text_with_colors', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_test.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube smoothed from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_smooth.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_smooth', RESAMBLEThreshold,
                        true,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_box', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube with non-uniformly spacing', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_non_uniformly_spaced.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_non_uniformly_spaced', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface smoothed from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_smooth.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface_smooth', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface smoothed with attributes from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_smooth_attributes.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface_smooth_attributes', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });
        it('Surface (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface_box', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw texture texts at cube vertices', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/cube_with_labels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'cube_with_labels', RESAMBLEThreshold,
                        true, done);
                }, true);


                self.K3D.load(json);
            });
        });

        it('should draw lines', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colors', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colors.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines_colors', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colors and mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colors_mesh.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines_colors_mesh', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colors and thick', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colors_thick.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines_colors_thick', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colors and mesh with proper joints', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colors_mesh_not_smooth.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines_colors_mesh_not_smooth',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colormap', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colormap.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines_colormap', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });


        it('should draw lines with colormap with mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colormap_mesh.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(
                        self.K3D, 'lines_colormap_mesh', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colormap with thick', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_colormap_thick.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(
                        self.K3D, 'lines_colormap_thick', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines with colormap with thick and axes helper', function (done) {
            var self = this;

            self.K3D.setAxesHelper(200);
            jsonLoader('http://localhost:9001/samples/lines_colormap_thick.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(
                        self.K3D, 'lines_colormap_thick_axes_helper', RESAMBLEThreshold, true, function () {
                            self.K3D.setAxesHelper(200);
                            done();
                        });
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw simple STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw simple smoothed STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_smooth.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl_smooth', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });
        it('should draw advanced STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_big.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl_big', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.setClearColor(0x000000);
                self.K3D.load(json);
            });
        });

        it('should draw a vector', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 0.5;

                self.K3D.load(json);
            });
        });

        it('should draw a text with LaTeX', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/latex.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'latex', RESAMBLEThreshold, false, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a label with LaTeX', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_with_annotations.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_with_annotations',
                        RESAMBLEThreshold, false, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 2d text with LaTeX', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/text2d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'text2d', RESAMBLEThreshold, false, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector with a label', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_label', RESAMBLEThreshold,
                        false, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 1.5;
                self.K3D.load(json);
            });
        });

        it('should draw a 2d vector field', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 2d vector field with a single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d_single_color',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });


        it('should draw a 2d vector field without heads', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d_no_head', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d_box', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field with single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field without heads', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d_no_head', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a texture', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/texture.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'texture', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a texture with data and colorLabel', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/texture_data.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'texture_data', RESAMBLEThreshold, false, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels with opacity', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_opacity.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_opacity', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels with opacity interior', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_opacity_interior.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_opacity_interior', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels with opacity and without grid', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_opacity.json', function (json) {

                self.K3D.setGridVisible(false);

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_opacity_without_grid', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_box', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels with outlines', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_outlines.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_outlines', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels with clipping planes', function (done) {
            var self = this;

            self.K3D.setClippingPlanes([
                [-1, -1.5, 0, 0],
                [0, 0, -1, 0.01]
            ]);

            jsonLoader('http://localhost:9001/samples/voxels.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_clipping_planes',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw sparse voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_sparse.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_sparse',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels_group', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_group.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_group',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw points with clipping planes', function (done) {
            var self = this;

            self.K3D.setClippingPlanes([
                [-1, 0, 0, 0]
            ]);

            jsonLoader('http://localhost:9001/samples/horse_with_colors_3d.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_3d_clipped',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a smoothed mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_smooth.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_smooth', RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_wireframe', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a opacity mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_opacity.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_opacity', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a mesh with face attribues', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_triangles_attribute.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_triangles_attribute', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe marching cubes', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_box_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_box_wireframe',
                        RESAMBLEThreshold, true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe stl', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl_wireframe', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_wireframe', RESAMBLEThreshold,
                        true, done);
                }, true);

                self.K3D.load(json);
            });
        });
    });
});
