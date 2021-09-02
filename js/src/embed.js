const _ = require('./lodash');

require('katex/dist/katex.min.css');
require('dat.gui/build/dat.gui.css');

module.exports = _.extend({}, require('./k3d'));
module.exports.version = require('./version').version;
