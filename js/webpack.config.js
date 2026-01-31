const CopyPlugin = require('copy-webpack-plugin');
const path = require('path');
const fs = require('fs');
const version = require('./package.json').version;
// var Visualizer = require('webpack-visualizer-plugin2');

// Custom webpack loaders are generally the same for all webpack bundles, hence
// stored in a separate local variable.
const rules = [
    {
        test: /\.(png|jpg|gif|svg|eot|ttf|woff|woff2)$/,
        type: 'asset/inline',
    },
    {
        test: /\.(glsl|txt)/,
        type: 'asset/source',
    },
    {
        resourceQuery: /raw/,
        type: 'asset/source',
    },
    // same as for jupyterlab packer
    // https://github.com/jupyterlab/jupyterlab/blob/3.1.x/builder/src/webpack.config.base.ts
    { test: /\.css$/, use: ['style-loader', 'css-loader'] },
];

const mode = 'production';

const plugins = [];

// plugins.push(new Visualizer({
//     filename: './webpack-statistics.html'
// }));

module.exports = [
    { // Notebook extension
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
            path: `${__dirname}/../k3d/static`,
            libraryTarget: 'amd',
        },
        module: {
            rules,
        },
        mode,
        plugins,
        externals: ['@jupyter-widgets/base', 'module'],
    },
    { // Bundle for the notebook containing the custom widget views and models
        //
        // This bundle contains the implementation for the custom widget views and
        // custom widget.
        // It must be an amd module
        //
        entry: ['./src/amd-public-path.js', './src/index.js'],
        output: {
            filename: 'index.js',
            path: `${__dirname}/../k3d/static`,
            libraryTarget: 'amd',
            publicPath: '', // Set in amd-public-path.js
        },
        mode,
        plugins,
        devtool: 'source-map',
        module: {
            rules,
        },
        // 'module' is the magic requirejs dependency used to set the publicPath
        externals: ['@jupyter-widgets/base', 'module'],
    },
    { // Embeddable k3d-jupyter bundle
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
        entry: ['./src/amd-public-path.js', './src/embed.js'],
        output: {
            filename: 'index.js',
            path: `${__dirname}/dist/`,
            libraryTarget: 'amd',
            publicPath: '', // Set in amd-public-path.js
        },
        mode,
        devtool: 'source-map',
        module: {
            rules,
        },
        plugins,
        // 'module' is the magic requirejs dependency used to set the publicPath
        externals: ['@jupyter-widgets/base', 'module'],
    },
    {
        entry: './src/standalone.js',
        output:
            {
                filename: 'standalone.js',
                path: `${__dirname}/../k3d/static`,
                library: 'k3d',
                libraryTarget: 'amd',
                publicPath: `https://unpkg.com/k3d@${version}/dist/`,
            },
        mode,
        devtool: 'source-map',
        module: {
            rules,
        },
        plugins: [
            new CopyPlugin({
                patterns: [
                    { from: './src/core/lib/headless.html' },
                    { from: './src/core/lib/snapshot_standalone.txt' },
                    { from: './src/core/lib/snapshot_online.txt' },
                    { from: './src/core/lib/snapshot_inline.txt' },
                    { from: './node_modules/requirejs/require.js' },
                    { from: './node_modules/fflate/umd/index.js', to: 'fflate.js' },
                    // Copy package.json to labextension folder
                    { from: './package.json', to: '../labextension' },
                ],
            }),
            // Custom plugin to copy built standalone.js to other locations
            {
                apply: (compiler) => {
                    compiler.hooks.afterEmit.tap('CopyBuildPlugin', (compilation) => {
                        const outputPath = compiler.options.output.path;
                        const files = ['standalone.js', 'standalone.js.map'];
                        const targets = [
                            path.resolve(__dirname, 'dist'),
                            path.resolve(__dirname, '../k3d/labextension/static')
                        ];
                        
                        targets.forEach(targetDir => {
                             if (!fs.existsSync(targetDir)) {
                                 fs.mkdirSync(targetDir, { recursive: true });
                             }
                        });

                        files.forEach(file => {
                            const src = path.join(outputPath, file);
                            if (fs.existsSync(src)) {
                                targets.forEach(targetDir => {
                                    fs.copyFileSync(src, path.join(targetDir, file));
                                });
                            }
                        });
                    });
                }
            }
        ],
    },
];
