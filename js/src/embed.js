var _ = require('./lodash');

require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!../node_modules/katex/dist/katex.min.css');
require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!../node_modules/dat.gui/build/dat.gui.css');

module.exports = _.extend({}, require('./k3d.js'));
module.exports.version = require('./version').version;
