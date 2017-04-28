'use strict';

/**
 * @memberof K3D.Config
 * @inner
 * @param  {Object} config  Current config object to find in
 * @return {String} path    A path to value being extracted
 */
function findByPath(config, path) {
    return path.split('.').reduce(function (value, key) {
        if (!value.hasOwnProperty(key)) {
            throw value;
        }

        return value[key];
    }, config);
}

/**
 * A generic config object used across K3D library
 * @memberof K3D
 * @method Config
 * @param {Object} config
 * @constructor
 */
function Config(config) {
    this.config = config;
}

/**
 * Extracts a value from K3D.Config#config object
 * @param {String} path
 * @param {*} defaultValue Default value returned if given path does not exist
 * @returns {*}
 */
Config.prototype.get = function (path, defaultValue) {
    try {
        return findByPath(this.config, path);
    } catch (e) {
        return defaultValue;
    }
};

/**
 * Checks if given path exists inside current K3D.Config#config object
 * @param {String} path
 * @returns {Boolean}
 */
Config.prototype.has = function (path) {
    try {
        findByPath(this.config, path);
    } catch (e) {
        return false;
    }

    return true;
};

module.exports = Config;
