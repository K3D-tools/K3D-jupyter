// Karma configuration
// Generated on Fri Oct 02 2015 12:10:30 GMT+0200 (CEST)

module.exports = function (config) {
    'use strict';

    config.set({

        // base path that will be used to resolve all patterns (eg. files, exclude)
        basePath: '',


        // frameworks to use
        // available frameworks: https://npmjs.org/browse/keyword/karma-adapter
        frameworks: ['jasmine'],


        // list of files / patterns to load in the browser
        files: [
            'node_modules/requirejs/require.js',
            'node_modules/karma-requirejs/lib/adapter.js',
            {pattern: 'dev/index.js', included: false},
            {pattern: 'node_modules/resemblejs/resemble.js', included: true},
            {pattern: 'test/utils/*.js', included: true},
            {pattern: 'node_modules/components-webfontloader/webfont.js', included: true},
            {pattern: 'test/assets/lato.css', included: true},
            {pattern: 'test/assets/style.css', included: true},
            {pattern: 'test/assets/Lato-Regular.ttf', included: false},
            {pattern: 'test/test-main.js', included: true},
            {pattern: 'test/**/*Spec.js', included: false}
        ],

        exclude: [
            'require.js'
        ],

        // test results reporter to use
        // possible values: 'dots', 'progress'
        // available reporters: https://npmjs.org/browse/keyword/karma-reporter
        reporters: ['spec'],

        // web server port
        port: 9876,

        // enable / disable colors in the output (reporters and logs)
        colors: true,

        // level of logging
        // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO ||
        // config.LOG_DEBUG
        logLevel: config.LOG_ERROR,

        // enable / disable watching file and executing tests whenever any file changes
        autoWatch: true,

        // start these browsers
        // available browser launchers: https://npmjs.org/browse/keyword/karma-launcher
        // browsers: ['Chrome'],
        browsers: ['Firefox'],
        browserNoActivityTimeout: 30000,
        // Continuous Integration mode
        // if true, Karma captures browsers, runs the tests and exits
        singleRun: true
    });
};
