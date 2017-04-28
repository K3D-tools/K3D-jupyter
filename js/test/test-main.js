var allTestFiles = [];
var TEST_REGEXP = /Spec\.js$/;

// Normalize a path to RequireJS module name.
var pathToModule = function (path) {
    return path.replace(/^\/base\//, '').replace(/\.js$/, '');
};

Object.keys(window.__karma__.files).forEach(function (file) {
    if (TEST_REGEXP.test(file)) {
        allTestFiles.push(pathToModule(file));
    }
});

require.config({
    // Karma serves files under `/base`,
    // which is the `basePath` from your config file.
    baseUrl: '/base',
    priority: ["libraries", "plugins"],

    paths: {
        K3D: 'dev/index'
    },
    shim: {},

    // Dynamically require all test files.
    deps: allTestFiles,

    // We have to kickoff testing framework,
    // after RequireJS is done with loading all the files.
    callback: function () {
        require(['K3D'], function () {
            jasmine.DEFAULT_TIMEOUT_INTERVAL = 30000;
            window.__karma__.start();
        });
    }
});


