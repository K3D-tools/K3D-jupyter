'use strict';

var FileSaver = require('file-saver');
var pako = require('pako');
var fileLoader = require('./helpers/fileLoader');
var template = require('raw-loader!./snapshot.txt');
var requireJsSource = require('raw-loader!./../../../node_modules/requirejs/require.js');
var pakoJsSource = require('raw-loader!./../../../node_modules/pako/dist/pako_inflate.min.js');
var semverRange = require('./../../version').version;

var sourceCode = window.k3dCompressed;
var scripts = document.getElementsByTagName('script');
var path;

if (typeof (sourceCode) === 'undefined') {
    sourceCode = '';

    for (var i = 0; i < scripts.length; i++) {
        // working in jupyter notebooks
        if (scripts[i].getAttribute('src') &&
            scripts[i].getAttribute('src').includes('k3d') &&
            scripts[i].getAttribute('src').includes('.js')) {
            path = scripts[i].getAttribute('src');
        }
    }

    if (typeof (path) !== 'undefined') {
        path = path.replace('k3d.js', 'standalone.js').replace('index.js', 'standalone.js');
    } else {
        // use npm repository
        path = 'https://unpkg.com/k3d@' + semverRange + '/dist/standalone.js';
    }

    fileLoader(path, function (data) {
        sourceCode = btoa(pako.deflate(data, {to: 'string', level: 9}));
    });
}

function getSnapshot(K3D) {
    var data = K3D.getSnapshot(),
        filecontent = template,
        timestamp = new Date().toUTCString();

    filecontent = filecontent.replace('[DATA]', btoa(data));
    filecontent = filecontent.replace('[PARAMS]', JSON.stringify(K3D.parameters));
    filecontent = filecontent.replace('[CAMERA]', JSON.stringify(K3D.getWorld().controls.getCameraArray()));
    filecontent = filecontent.replace('[TIMESTAMP]', timestamp);
    filecontent = filecontent.replace('[REQUIRE_JS]', requireJsSource);
    filecontent = filecontent.replace('[PAKO_JS]', pakoJsSource);
    filecontent = filecontent.replace('[K3D_SOURCE]', sourceCode);

    return filecontent;
}

function handleFileSelect(K3D, evt) {
    var files = evt.dataTransfer.files,
        reader = new FileReader();

    evt.stopPropagation();
    evt.preventDefault();

    reader.onload = function (event) {
        var snapshot = K3D.extractSnapshot(event.target.result);

        if (snapshot[1]) {
            K3D.setSnapshot(atob(snapshot[1]));
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
                var data = getSnapshot(K3D),
                    filename = 'K3D-snapshot-' + Date.now() + '.html';

                if (K3D.parameters.name) {
                    filename = K3D.parameters.name + '.html';
                }

                data = new Blob([data], {type: 'text/plain;charset=utf-8'});
                FileSaver.saveAs(data, filename);
            }
        },
        targetDomNode;

    gui.add(obj, 'snapshot').name('Snapshot HTML');

    // Setup the dnd listeners.
    targetDomNode = K3D.getWorld().targetDOMNode;

    targetDomNode.addEventListener('dragover', handleDragOver, false);
    targetDomNode.addEventListener('drop', handleFileSelect.bind(null, K3D), false);
}

module.exports = {
    snapshotGUI: snapshotGUI,
    getSnapshot: getSnapshot
};
