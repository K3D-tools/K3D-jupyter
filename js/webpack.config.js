var version = require('./package.json').version;
var webpack = require('webpack');

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

var plugins = [];

plugins.push(new webpack.optimize.UglifyJsPlugin({
    compress: {warnings: false},
    sourceMap: true
}));

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
        plugins: plugins,
        devtool: 'source-map',
        module: {
            rules: rules
        },
        externals: ['jupyter-js-widgets']
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
            library: "K3D",
            libraryTarget: 'amd',
            publicPath: 'https://unpkg.com/K3D@' + version + '/dist/'
        },
        devtool: 'source-map',
        module: {
            rules: rules
        },
        plugins: plugins,
        externals: ['jupyter-js-widgets']
    }
];
