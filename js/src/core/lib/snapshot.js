'use strict';
var FileSaver = require('file-saver');
var template = require('raw-loader!./snapshot.html');

function getSnapshot(K3D) {
    var data = K3D.getSnapshot();
    var filecontent = template;
    var timestamp = new Date().toUTCString();

    filecontent = filecontent.replace('[JS_VERSION]', K3D.parameters.guiVersion);
    filecontent = filecontent.replace('[TIMESTAMP]', timestamp);
    filecontent = filecontent.replace('[DATA]', btoa(data));

    return filecontent;
}

function snapshotGUI(gui, K3D) {
    var obj = {
        snapshot: function () {
            var data = getSnapshot(K3D);

            data = new Blob([data], {type: 'text/plain;charset=utf-8'});
            FileSaver.saveAs(data, 'K3D-snapshot-' + Date.now() + '.html');
        }
    };

    gui.add(obj, 'snapshot').name('Snapshot HTML');
}

module.exports = {
    snapshotGUI: snapshotGUI,
    getSnapshot: getSnapshot
};
