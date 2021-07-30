/* jshint indent: false, quotmark: false */

const webpackConfig = require('./webpack.config');

module.exports = function (grunt) {
    require('time-grunt')(grunt);
    require('load-grunt-tasks')(grunt);

    grunt.initConfig({
        webpack: {
            myConfig: webpackConfig
        },
        eslint: {
            options: {
                config: '.eslintrc.js'
            },
            target: ['src/core/*.js']
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
                    livereload: true
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
        'eslint'
    ]);

    grunt.registerTask('doc', [
        'clean:doc',
        'jsdoc'
    ]);

    grunt.registerTask('test', () => {
        grunt.task.run([
            'clean',
            'webpack',
            'express:test',
            'curl',
            'karma'
        ]);
    });

    grunt.registerTask('build', () => {
        grunt.task.run([
            'clean',
            'webpack'
        ]);
    });

    grunt.registerTask('serve', () => {
        grunt.task.run([
            'clean',
            'webpack',
            'connect',
            'open:dev',
            'watch:webpack'
        ]);
    });
};
