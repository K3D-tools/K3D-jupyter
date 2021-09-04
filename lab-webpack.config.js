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
    }
};
