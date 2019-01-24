'use strict';

module.exports = {
    diff: function (config, prevConfig) {
        var diff = {};

        Object.keys(config).forEach(function (key) {
            if (config[key] !== prevConfig[key]) {
                diff[key] = true;
            }
        });

        return diff;
    }
};
