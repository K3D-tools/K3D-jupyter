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

function handleFileSelect(K3D, evt) {
    evt.stopPropagation();
    evt.preventDefault();

    var files = evt.dataTransfer.files;
    var reader = new FileReader();

    reader.onload = function (event) {
        var snapshot = K3D.extractSnapshot(event.target.result);

        if(snapshot[1]) {
            K3DInstance.setSnapshot(atob(snapshot[1]));
        }
    };

    if (files.length > 0) {
        reader.readAsText(files[0]);
    }
}

function handleDragOver(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
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

    // Setup the dnd listeners.
    var targetDomNode = K3D.getWorld().targetDOMNode;

    targetDomNode.addEventListener('dragover', handleDragOver, false);
    targetDomNode.addEventListener('drop', handleFileSelect.bind(this, K3D), false);
}

module.exports = {
    snapshotGUI: snapshotGUI,
    getSnapshot: getSnapshot
};
