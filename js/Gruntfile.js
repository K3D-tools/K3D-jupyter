/* jshint indent: false, quotmark: false */

const webpackConfig = require('./webpack.config');

module.exports = function (grunt) {
    require('time-grunt')(grunt);
    require('load-grunt-tasks')(grunt);

    grunt.initConfig({
        webpack: {
            myConfig: webpackConfig,
        },
        eslint: {
            options: {
                config: '.eslintrc.js',
            },
            target: ['src/core/*.js'],
        },
        watch: {
            webpack: {
                files: [
                    'src/**/*.js',
                    'src/**/*.glsl',
                    'src/**/*.css',
                ],
                tasks: ['webpack'],
                options: {
                    livereload: true,
                },
            },
            development: {
                files: [
                    'development.html',
                ],
                options: {
                    livereload: true,
                },
            },
        },
        connect: {
            server: {
                options: {
                    port: 9000,
                    base: './',
                },
            },
        },
        jsdoc: {
            dist: {
                src: ['src/providers/**/*.js', 'src/core/**/*.js'],
                options: {
                    destination: 'doc',
                    readme: 'README.md',
                },
            },
        },
        open: {
            dev: {
                path: 'http://localhost:9000/development.html',
            },
        },
        clean: {
            doc: 'doc',
            dist: 'dist',
            dev: 'dev',
        },
    });

    grunt.registerTask('codeStyle', [
        'eslint',
    ]);

    grunt.registerTask('doc', [
        'clean:doc',
        'jsdoc',
    ]);

    grunt.registerTask('build', () => {
        grunt.task.run([
            'clean',
            'webpack',
        ]);
    });

    grunt.registerTask('serve', () => {
        grunt.task.run([
            'clean',
            'webpack',
            'connect',
            'open:dev',
            'watch',
        ]);
    });
};
