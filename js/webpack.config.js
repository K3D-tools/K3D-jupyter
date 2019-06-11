var version = require('./package.json').version;
var webpack = require('webpack');
var nodeExternals = require('webpack-node-externals');
// var Visualizer = require('webpack-visualizer-plugin');

// Custom webpack loaders are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [
    {
        test: /\.(png|jpg|gif|svg|eot|ttf|woff|woff2)$/,
        use: 'url-loader'
    },
    {
        test: /\.glsl/,
        use: 'raw-loader'
    }
];

var mode = 'production';

var plugins = [];

plugins.push(new webpack.optimize.AggressiveMergingPlugin());

// plugins.push(new Visualizer({
//     filename: './webpack-statistics.html'
// }));

module.exports = [
    {// Notebook extension
        //
        // This bundle only contains the part of the JavaScript that is run on
        // load of the notebook. This section generally only performs
        // some configuration for requirejs, and provides the legacy
        // "load_ipython_extension" function which is required for any notebook
        // extension.
        //
        entry: './src/extension.js',
        output: {
            filename: 'extension.js',
            path: __dirname + '/../k3d/static',
            libraryTarget: 'amd'
        },
        module: {
            rules: rules
        },
        mode: mode,
        plugins: plugins
    },
    {// Bundle for the notebook containing the custom widget views and models
        //
        // This bundle contains the implementation for the custom widget views and
        // custom widget.
        // It must be an amd module
        //
        entry: './src/index.js',
        output: {
            filename: 'index.js',
            path: __dirname + '/../k3d/static',
            libraryTarget: 'amd'
        },
        mode: mode,
        plugins: plugins,
        devtool: 'source-map',
        module: {
            rules: rules
        },
        externals: ['@jupyter-widgets/base']
    },
    {// Embeddable K3D-jupyter bundle
        //
        // This bundle is generally almost identical to the notebook bundle
        // containing the custom widget views and models.
        //
        // The only difference is in the configuration of the webpack public path
        // for the static assets.
        //
        // It will be automatically distributed by unpkg to work with the static
        // widget embedder.
        //
        // The target bundle is always `dist/index.js`, which is the path required
        // by the custom widget embedder.
        //
        entry: './src/embed.js',
        output: {
            filename: 'index.js',
            path: __dirname + '/dist/',
            libraryTarget: 'amd',
            publicPath: 'https://unpkg.com/k3d@' + version + '/dist/'
        },
        mode: mode,
        devtool: 'source-map',
        module: {
            rules: rules
        },
        plugins: plugins,
        externals: ['@jupyter-widgets/base']
    },
    {
        entry: './src/standalone.js',
        output:
            {
                filename: 'standalone.js',
                path: __dirname + '/../k3d/static',
                library: "k3d",
                libraryTarget: 'amd',
                publicPath: 'https://unpkg.com/k3d@' + version + '/dist/'
            },
        mode: mode,
        devtool: 'source-map',
        module: {
            rules: rules
        },
        plugins: plugins
    },
    {
        entry: './src/standalone.js',
        output:
            {
                filename: 'standalone.js',
                path: __dirname + '/dist/',
                library: "k3d",
                libraryTarget: 'amd',
                publicPath: 'https://unpkg.com/k3d@' + version + '/dist/'
            },
        mode: mode,
        devtool: 'source-map',
        module: {
            rules: rules
        },
        plugins: plugins
    },
    {  // Lab extension is just our JS source + shaders
        entry: './src/labplugin.js',
        output: {
            filename: 'labplugin.js',
            path: __dirname + '/dist/',
            libraryTarget: 'amd'
        },
        mode: mode,
        devtool: 'source-map',
        module: {
            rules: [
                {
                    test: /\.glsl/,
                    use: 'raw-loader'
                }
            ]
        },
        externals: [nodeExternals()]
    }
];
