'use strict';

var FileSaver = require('file-saver');
var pako = require('pako');
var fileLoader = require('./helpers/fileLoader');
var templateStandalone = require('./snapshot_standalone.txt');
var templateOnline = require('./snapshot_online.txt');
var requireJsSource = require('./../../../node_modules/requirejs/require.js?raw');
var pakoJsSource = require('./../../../node_modules/pako/dist/pako_inflate.min.js?raw');
var semverRange = require('./../../version').version;
var buffer = require('./helpers/buffer');

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
        sourceCode = buffer.arrayBufferToBase64(pako.deflate(data));
    });
}

function getHTMLSnapshot(K3D, compressionLevel) {
    var data = buffer.arrayBufferToBase64(K3D.getSnapshot(compressionLevel)),
        filecontent,
        timestamp = new Date().toUTCString();

    if (K3D.parameters.snapshotIncludeJs) {
        filecontent = templateStandalone;
        filecontent = filecontent.split('[REQUIRE_JS]').join(requireJsSource);
        filecontent = filecontent.split('[PAKO_JS]').join(pakoJsSource);
        filecontent = filecontent.split('[K3D_SOURCE]').join(sourceCode);
    } else {
        filecontent = templateOnline;
        filecontent = filecontent.split('[VERSION]').join(K3D.parameters.guiVersion);
    }

    filecontent = filecontent.split('[DATA]').join(data);
    filecontent = filecontent.split('[PARAMS]').join(JSON.stringify(K3D.parameters));
    filecontent = filecontent.split('[CAMERA]').join(JSON.stringify(K3D.getWorld().controls.getCameraArray()));
    filecontent = filecontent.split('[TIMESTAMP]').join(timestamp);
    filecontent = filecontent.split('[ADDITIONAL]').join('//[ADDITIONAL]');

    return filecontent;
}

function handleFileSelect(K3D, evt) {
    var files = evt.dataTransfer.files,
        snapshotReader = new FileReader(),
        STLReader = new FileReader();

    evt.stopPropagation();
    evt.preventDefault();

    snapshotReader.onload = function (event) {
        var snapshot = K3D.extractSnapshot(event.target.result);

        if (snapshot[1]) {
            K3D.setSnapshot(snapshot[1]);
        }
    };

    STLReader.onload = function (event) {
        var stl = event.target.result;

        K3D.load({
            objects: [{
                type: 'STL',
                model_matrix: {
                    data: [
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1]
                },
                binary: stl,
                visible: true,
                color: 0x0000ff,
                wireframe: false,
                flat_shading: false
            }]
        });
    };

    if (files.length > 0) {
        if (files[0].name.substr(-4).toLowerCase() === 'html') {
            snapshotReader.readAsText(files[0]);
        }

        if (files[0].name.substr(-3).toLowerCase() === 'stl') {
            STLReader.readAsArrayBuffer(files[0]);
        }
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
                var data = getHTMLSnapshot(K3D, 9),
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
    getHTMLSnapshot: getHTMLSnapshot
};
