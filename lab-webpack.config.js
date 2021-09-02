const version = require('./package.json').version;

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

const plugins = [];



module.exports = {
    module: {
        rules,
    }
};
