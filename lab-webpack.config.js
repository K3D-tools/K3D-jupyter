const path = require('path');

const rules = [
    {
        test: /\.(glsl|txt)/,
        type: 'asset/source',
    }
    ,
    {
        resourceQuery: /raw/,
        type: 'asset/source',
    },
];

module.exports = {
    module: {
        rules,
    },
    resolve: {
        alias: {
            'lil-gui': path.resolve(__dirname, 'js/node_modules/lil-gui/dist/lil-gui.esm.js'),
        },
    },
};
