/* jshint indent: false, quotmark: false */
'use strict';

var webpackConfig = require('./webpack.config');

module.exports = function (grunt) {

    require('time-grunt')(grunt);
    require('load-grunt-tasks')(grunt);

    grunt.initConfig({
        webpack: {
            myConfig: webpackConfig,
        },
        jshint: {
            options: {
                jshintrc: '.jshintrc',
                reporter: require('jshint-stylish'),
                reporterOutput: ''
            },
            tools: ['Gruntfile.js', 'expressTestHelper.js'],
            k3d: ['src/**/*.js']
        },
        jscs: {
            options: {
                config: '.jscsrc'
            },
            tools: ['Gruntfile.js', 'expressTestHelper.js'],
            k3d: ['src/**/*.js']
        },
        watch: {
            webpack: {
                files: [
                    'src/**/*.js',
                    'src/**/*.glsl',
                    'src/**/*.css',
                    'development.html'
                ],
                tasks: ['webpack'],
                options: {
                    livereload: true,
                }
            }
        },
        connect: {
            server: {
                options: {
                    port: 9000,
                    base: './'
                }
            }
        },
        jsdoc: {
            dist: {
                src: ['src/providers/**/*.js', 'src/core/**/*.js'],
                options: {
                    destination: 'doc',
                    readme: 'README.md'
                }
            }
        },
        karma: {
            unit: {
                configFile: 'karma.conf.js'
            }
        },
        open: {
            dev: {
                path: 'http://localhost:9000/development.html'
            }
        },
        express: {
            test: {
                options: {
                    script: 'expressTestHelper.js'
                }
            }
        },
        clean: {
            test: 'src/test/results/*.png',
            doc: 'doc',
            dist: 'dist',
            dev: 'dev'
        },
        curl: {
            'test/assets/Lato-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf'
        }
    });

    grunt.registerTask('codeStyle', [
        'jshint',
        'jscs'
    ]);

    grunt.registerTask('doc', [
        'clean:doc',
        'jsdoc'
    ]);

    grunt.registerTask('test', function () {
        grunt.task.run([
            'clean',
            'webpack',
            'express:test',
            'curl',
            'karma'
        ]);
    });

    grunt.registerTask('build', function () {
        grunt.task.run([
            'clean',
            'webpack'
        ]);
    });

    grunt.registerTask('serve', function () {
        grunt.task.run([
            'clean',
            'webpack',
            'connect',
            'open:dev',
            'watch:webpack'
        ]);
    });
};
