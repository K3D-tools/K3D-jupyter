var allTestFiles = [];
var TEST_REGEXP = /Spec\.js$/;

// Get a list of all the test files to include
Object.keys(window.__karma__.files).forEach(function (file) {
    if (TEST_REGEXP.test(file)) {
        // Normalize paths to RequireJS module names.
        // If you require sub-dependencies of test files to be loaded as-is (requiring file extension)
        // then do not normalize the paths
        var normalizedTestModule = file.replace(/^\/base\/|\.js$/g, '')
        allTestFiles.push(normalizedTestModule)
    }
});

require.config({
    // Karma serves files under `/base`,
    // which is the `basePath` from your config file.
    baseUrl: '/base',
    priority: ["libraries", "plugins"],

    paths: {
        K3D: 'dist/standalone.js'
    },
    shim: {},

    // Dynamically require all test files.
    deps: allTestFiles,

    // We have to kickoff testing framework,
    // after RequireJS is done with loading all the files.
    callback: function () {
        require(['k3d'], function () {
            jasmine.DEFAULT_TIMEOUT_INTERVAL = 30000;
            window.__karma__.start();
        });
    }
});


