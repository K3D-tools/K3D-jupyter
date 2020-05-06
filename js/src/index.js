var _ = require('./lodash');

// Export widget models and views, and the npm package version number.
module.exports = _.extend({}, require('./k3d.js'));
module.exports.version = require('./version').version;
