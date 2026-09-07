const webpackConfig = require('./webpack.config');

module.exports = function (grunt) {
    require('time-grunt')(grunt);
    require('load-grunt-tasks')(grunt);

    grunt.initConfig({
        webpack: {
            myConfig: webpackConfig,
        },
        eslint: {
            // ESLint constructor options; left empty so eslint finds eslint.config.js on its own
            options: {},
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

    // the project's own eslint through its Node API: one eslint version, no grunt plugin
    grunt.registerTask('eslint', 'Lint the sources with eslint', function () {
        const done = this.async();
        const { ESLint } = require('eslint');
        const eslint = new ESLint(grunt.config('eslint.options'));

        eslint.lintFiles(grunt.config('eslint.target'))
            .then((results) => {
                // printed by hand: the stylish formatter colours through util.styleText,
                // which Node below 22.13 lacks, and the plain formatters left core in eslint 9
                results.forEach((result) => {
                    result.messages.forEach((m) => {
                        const level = m.severity === 2 ? 'error' : 'warning';

                        grunt.log.writeln(
                            `${result.filePath}:${m.line}:${m.column} ${level} ${m.message} (${m.ruleId})`,
                        );
                    });
                });

                done(results.every((r) => r.errorCount === 0));
            })
            .catch((error) => {
                grunt.log.error(error.message);
                done(false);
            });
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
            'watch',
        ]);
    });
};
