// Import package data to define it only one place
var pkg = require('../package.json');

/**
 * The version of the attribute spec that this package
 * implements. This is the value used in
 * _model_module_version/_view_module_version.
 *
 * Update this value when attributes are added/removed from
 * your models, or serialized format changes.
 */
var EXTENSION_SPEC_VERSION = '3.0.0';

module.exports = {
    version: pkg.version,
    EXTENSION_SPEC_VERSION: EXTENSION_SPEC_VERSION
};
