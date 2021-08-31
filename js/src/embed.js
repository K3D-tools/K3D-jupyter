const _ = require('./lodash');

// require('style-loader?{attributes:{id: "k3d-katex"}}!css-loader!katex/dist/katex.min.css');
// require('style-loader?{attributes:{id: "k3d-dat.gui"}}!css-loader!dat.gui/build/dat.gui.css');

require('katex/dist/katex.min.css');
require('dat.gui/build/dat.gui.css');

module.exports = _.extend({}, require('./k3d'));
module.exports.version = require('./version').version;
