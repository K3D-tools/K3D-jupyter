require('es6-promise');

module.exports = {
    K3D: require('./core/Core'),
    Config: require('./core/lib/Config'),
    ThreeJsProvider: require('./providers/threejs/provider')
};
