define(['K3D'], function (lib) {
    var K3D = lib.K3D;
    var ThreeJsProvider = lib.ThreeJsProvider;

    const RESAMBLEThreshold = 0.25;

    function jsonLoader(url, callback) {

        var xhrLoad = new XMLHttpRequest();

        xhrLoad.open('GET', url, true);

        xhrLoad.onreadystatechange = function () {
            if (xhrLoad.readyState === 4) {
                callback(JSON.parse(xhrLoad.response));
            }
        };

        xhrLoad.send(null);
    }

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
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'torus-knots', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model without colors should be drawn (base64)', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_without_colors_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'horse_without_colors',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model without colors should be drawn (transferred without base64)', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'horse_with_colors', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Points-based horse model with colors should be drawn (base64)', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/horse_with_colors_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'horse_with_colors_base64',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple single line text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_text_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'simple_text_label', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Simple multiline text label should be drawn', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/simple_multiline_text_label.json', function (json) {
                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'simple_multiline_text_label',
                        RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a text with fill and stroke colors', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/text_with_colors.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'text_with_colors', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_test.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'marching_cubes', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'marching_cubes_box', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Marching cube from base64 array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/marching_cubes_test_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'marching_cubes', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'surface', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface (not cube) from normal array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'surface_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Surface from base64 array', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/surface_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'surface', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('Cube with described vertices', function (done) {

            var self = this;

            jsonLoader('http://localhost:9001/samples/cube_with_labels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'cube_with_labels', RESAMBLEThreshold,
                        done);
                }, true);


                self.K3D.load(json);
            });
        });

        it('should draw lines', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'lines', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw lines (base64)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/lines_base64.json', function (json) {

                self.K3D.getWorld().camera.position.z = 15;

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'lines', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw simple STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'stl', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw advanced STL geometry', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/stl_big.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'stl_big', RESAMBLEThreshold, done);
                }, true);

                self.K3D.setClearColor(0x000000, 1);
                self.K3D.load(json);
            });
        });

        it('should draw a vector', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 0.5;

                self.K3D.load(json);
            });
        });

        it('should draw a vector (base64)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vectors_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 0.5;

                self.K3D.load(json);
            });
        });

        it('should draw a vector with a label', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector_label.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector_label', RESAMBLEThreshold, done);
                }, true);

                self.K3D.getWorld().camera.position.z = 1.5;
                self.K3D.load(json);
            });
        });

        it('should draw a vector field 2d', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector2d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 2d with single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector2d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });


        it('should draw a vector field 2d without head', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector2d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector2d_no_head', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 2d (base64)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector2d_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector2d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 3d', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector3d.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector3d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 3d (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector3d_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector3d_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 3d (base64)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector3d_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector3d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 3d with single color', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector3d_single_color.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector3d', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a vector field 3d without head', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/vector3d_no_head.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'vector3d_no_head', RESAMBLEThreshold,
                        done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a texture', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/texture.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'texture', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a voxels', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'voxels', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a voxels (not cube)', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_box.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'voxels_box', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });

        it('should draw a voxels from base64 array', function (done) {
            var self = this;

            jsonLoader('http://localhost:9001/samples/voxels_base64.json', function (json) {

                self.K3D.addFrameUpdateListener('after', function () {
                    TestHelpers.compareCanvasWithExpectedImage(self.canvas.id, 'voxels', RESAMBLEThreshold, done);
                }, true);

                self.K3D.load(json);
            });
        });
    });
});