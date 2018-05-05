require(['K3D'], function (lib) {
    var K3D = lib.K3D;
    var ThreeJsProvider = lib.ThreeJsProvider;
    var jsonLoader = TestHelpers.jsonLoader;

    const RESAMBLEThreshold = 0.2;

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
            window.K3DInstance = this.K3D = K3D(ThreeJsProvider, this.canvas, {antialias: false});
        });

        afterEach(function () {
            this.K3D.disable();
            TestHelpers.removeTestCanvas(this.canvas);
        });

        it('2 torus knots should be drawn as on reference image', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/torus-knots.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'torus-knots', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model without colors should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_without_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_without_colors',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=3d should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_3d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_3d',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=flat should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_flat.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_flat',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors and shader=mesh should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_mesh.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'horse_with_colors_mesh',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple single line text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_text_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'simple_text_label', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple multiline text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_multiline_text_label.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'simple_multiline_text_label',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a text with fill and stroke colors', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/text_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'text_with_colors', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_test.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'surface_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw texture texts at cube vertices', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/cube_with_labels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'cube_with_labels', RESAMBLEThreshold, done);
                }, true);


                self.K3D.load(json);
            });
        });

        it('should draw lines', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'lines', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw simple STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw advanced STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_big.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl_big', RESAMBLEThreshold, done);
                }, true);

                self.K3D.setClearColor(0x000000, 1);
                self.K3D.load(json);
            });
        });

        it('should draw a vector', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 0.5;

                self.K3D.load(json);
            });
        });

        it('should draw a text with LaTeX', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/latex.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'latex', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 1.5;
                self.K3D.load(json);
            });
        });

        it('should draw a 2d text with LaTeX', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/text2d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'text2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector with a label', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_label', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 1.5;
                self.K3D.load(json);
            });
        });

        it('should draw a 2d vector field', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 2d vector field with a single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });


        it('should draw a 2d vector field without heads', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_2d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_2d_no_head', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field with single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a 3d vector field without heads', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_field_3d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'vector_field_3d_no_head', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a texture', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/texture.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'texture', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw voxels (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe mesh', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/mesh_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'mesh_wireframe', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe marching cubes', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_box_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'marching_cubes_box_wireframe',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe stl', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'stl_wireframe', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a wireframe voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_wireframe.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.K3D, 'voxels_wireframe', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });
    });
});
