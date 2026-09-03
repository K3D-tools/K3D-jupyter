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
                // left empty: eslint finds eslint.config.js in this directory on its own
            },
            // the sources plus the two build files - nothing in js/ is left unlinted
            target: ['src/**/*.js', 'Gruntfile.js', 'webpack.config.js'],
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
        open: {
            dev: {
                path: 'http://localhost:9000/development.html',
            },
        },
        clean: {
            dist: 'dist',
            dev: 'dev',
        },
    });

    grunt.registerTask('codeStyle', [
        'eslint',
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
