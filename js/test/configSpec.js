'use strict';

define(['K3D'], function (lib) {
    var Config = lib.Config;

    describe('Config', function () {
        beforeAll(function () {
            this.config = new Config({
                x: 1,
                y: {
                    a: ['X', 'Y']
                }
            });
        });

        describe('get', function () {
            it('should return defined value for existing key', function () {
                expect(this.config.get('x')).toEqual(1);
            });

            it('should return default value for non existing key', function () {
                expect(this.config.get('z', 'default')).toEqual('default');
            });

            it('should return defined value for existing nested key', function () {
                expect(this.config.get('y.a')).toEqual(['X', 'Y']);
            });

            it('should return default value for non existing nested key', function () {
                expect(this.config.get('y.b', 'default')).toEqual('default');
            });
        });

        describe('has', function () {
            it('should return true for existing key', function () {
                expect(this.config.has('x')).toEqual(true);
            });

            it('should return false for non existing key', function () {
                expect(this.config.has('z')).toEqual(false);
            });

            it('should return true for existing nested key', function () {
                expect(this.config.has('y.a')).toEqual(true);
            });

            it('should return false for non existing nested key', function () {
                expect(this.config.has('y.b')).toEqual(false);
            });
        });
    });
});