// Flat config, because eslint 10 removed the .eslintrc format outright.
const js = require('@eslint/js');
const globals = require('globals');
const { FlatCompat } = require('@eslint/eslintrc');

const compat = new FlatCompat({
    baseDirectory: __dirname,
    recommendedConfig: js.configs.recommended,
});

module.exports = [
    // airbnb-base is consumed through FlatCompat rather than transcribed: it stays the one
    // source of the rule set, so nothing can be dropped from it by hand.
    ...compat.extends('airbnb-base'),
    {
        languageOptions: {
            // flat config has no `env`: browser and es2020 came from .eslintrc.js, node from airbnb
            globals: {
                ...globals.browser,
                ...globals.node,
                ...globals.es2020,
            },
        },
        rules: {
            indent: ['error', 4, { SwitchCase: 1 }],
            'linebreak-style': 0,
            'max-len': ['error', 120],
            'import/no-webpack-loader-syntax': 0,
            'import/no-unresolved': 0,
            // deep imports into three / three-gpu-pathtracer / three-mesh-bvh sources are not
            // package exports, so the .js extension is mandatory there
            'import/extensions': ['error', 'ignorePackages', { js: 'ignorePackages' }],
            'func-names': 0,
            'no-underscore-dangle': 0,
            'global-require': 0,
            'no-param-reassign': 0,
            'prefer-rest-params': 0,
            'prefer-spread': 0,
            'no-plusplus': 0,
            'no-bitwise': 0,
            'no-continue': 0,
            'no-console': 0,
            'prefer-destructuring': 0,
            'no-prototype-builtins': 0,
            'no-restricted-properties': 0,
            'no-use-before-define': 0,
            'no-loop-func': 0,
            // K3D is the single ambient object; helpers take it explicitly even when the
            // enclosing Init(K3D) scope already has it, because other modules call them
            'no-shadow': ['error', { allow: ['K3D'] }],
            // airbnb's three options, plus eslint 8's caughtErrors: the five empty catches in
            // Core.js and webglBackend.js are deliberate, and ecmaVersion 2018 has no `catch {}`
            'no-unused-vars': ['error', {
                vars: 'all',
                args: 'after-used',
                ignoreRestSiblings: true,
                caughtErrors: 'none',
            }],
        },
    },
];
