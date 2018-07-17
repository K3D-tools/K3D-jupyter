// Entry point for the unpkg bundle containing custom model definitions.
//
// It differs from the notebook bundle in that it does not need to define a
// dynamic baseURL for the static assets and may load some css that would
// already be loaded by the notebook otherwise.

// Export widget models and views, and the npm package version number.
var _ = require('lodash');

require('style-loader?{attrs:{id: "k3d-katex"}}!css-loader!../node_modules/katex/dist/katex.min.css');
require('style-loader?{attrs:{id: "k3d-dat.gui"}}!css-loader!../node_modules/dat.gui/build/dat.gui.css');

module.exports = _.extend({}, require('./k3d.js'));
module.exports.version = require('./version').version;
