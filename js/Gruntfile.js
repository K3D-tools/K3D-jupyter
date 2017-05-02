/* jshint indent: false, quotmark: false */
'use strict';

const webpackConfig = require('./webpack.config');
const webpackConfigDev = require('./webpack.config.dev');

var LIVERELOAD_PORT = 35729,
    compression = require('compression'),
    lrSnippet = require('connect-livereload')({
        port: LIVERELOAD_PORT
    }),

    mountFolder = function (connect, dir) {
        return connect.static(require('path').resolve(dir));
    };

module.exports = function (grunt) {

    require('time-grunt')(grunt);
    require('load-grunt-tasks')(grunt);

    grunt.initConfig({
        webpack: {
            prod: webpackConfig,
            dev: webpackConfigDev
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
            livereload: {
                options: {
                    livereload: LIVERELOAD_PORT
                },
                files: [
                    'src/**/*.js',
                    'src/**/*.css',
                    'development.html'
                ]
            },
            // jsdoc: {
            //     files: ['src/**/*.js'],
            //     tasks: ['jsdoc']
            // },
            webpack: {
                files: ['src/**/*.js'],
                tasks: ['webpack:dev']
            }
        },
        connect: {
            options: {
                port: 9000,
                // change this to '0.0.0.0' to access the server from outside
                hostname: '0.0.0.0'
            },
            livereload: {
                options: {
                    middleware: function (connect) {
                        return [
                            compression(),
                            lrSnippet,
                            mountFolder(connect, './')
                        ];
                    }
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
                path: 'http://localhost:<%= connect.options.port %>/development.html'
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
            'test/assets/Lato-Regular.ttf': 'https://github.com/google/fonts/raw/master/ofl/lato/Lato-Regular.ttf'
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
            'webpack:prod'
        ]);
    });

    grunt.registerTask('serve', function () {
        grunt.task.run([
            'clean',
            'webpack:dev',
            'connect:livereload',
            'open:dev',
            'watch'
        ]);
    });
};
