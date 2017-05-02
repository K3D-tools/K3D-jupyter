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

module.exports = [
    {
        entry: './src/development.js',
        output: {
            filename: 'index.js',
            path: __dirname + '/dev/',
            library: "K3D",
            libraryTarget: 'amd'
        },
        devtool: 'source-map',
        module: {
            rules: rules
        }
    }
];
