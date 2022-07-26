const FileSaver = require('file-saver');
const fflate = require('fflate');
const requireJsSource = require('requirejs/require?raw');
const fflateJsSource = require('../../../node_modules/fflate/umd/index?raw');
const fileLoader = require('./helpers/fileLoader');
const templateStandalone = require('./snapshot_standalone.txt');
const templateOnline = require('./snapshot_online.txt');
const templateInline = require('./snapshot_inline.txt');
const semverRange = require('../../version').version;
const buffer = require('./helpers/buffer');

let sourceCode = window.k3dCompressed;
const scripts = document.getElementsByTagName('script');
let path;

if (typeof (sourceCode) === 'undefined') {
    sourceCode = '';

    for (let i = 0; i < scripts.length; i++) {
        // working in jupyter notebooks
        if (scripts[i].getAttribute('src')
            && scripts[i].getAttribute('src').includes('k3d')
            && scripts[i].getAttribute('src').includes('.js')) {
            path = scripts[i].getAttribute('src');
        }
    }

    if (typeof (path) !== 'undefined') {
        path = path.replace('k3d.js', 'standalone.js').replace('index.js', 'standalone.js');
    } else {
        // use npm repository
        path = `https://unpkg.com/k3d@${semverRange}/dist/standalone.js`;
    }

    fileLoader(path, (data) => {
        data = fflate.strToU8(data);
        sourceCode = buffer.arrayBufferToBase64(fflate.zlibSync(data));
    });
}

function getHTMLSnapshot(K3D, compressionLevel) {
    K3D.heavyOperationSync = true;

    const data = buffer.arrayBufferToBase64(K3D.getSnapshot(compressionLevel));
    let filecontent;
    const timestamp = new Date().toUTCString();

    if (K3D.parameters.snapshotType === 'full') {
        filecontent = templateStandalone;
        filecontent = filecontent.split('[REQUIRE_JS]').join(requireJsSource);
        filecontent = filecontent.split('[FFLATE_JS]').join(fflateJsSource);
        filecontent = filecontent.split('[K3D_SOURCE]').join(sourceCode);
    } else if (K3D.parameters.snapshotType === 'online') {
        filecontent = templateOnline;
        filecontent = filecontent.split('[VERSION]').join(K3D.parameters.guiVersion);
    } else if (K3D.parameters.snapshotType === 'inline') {
        filecontent = templateInline;
        filecontent = filecontent.split('[ID]').join(Math.floor(Math.random() * 1e9));
        filecontent = filecontent.split('[HEIGHT]').join(K3D.parameters.height || 512);
        filecontent = filecontent.split('[VERSION]').join(K3D.parameters.guiVersion);
    }

    filecontent = filecontent.split('[DATA]').join(data);
    filecontent = filecontent.split('[TIMESTAMP]').join(timestamp);
    filecontent = filecontent.split('[ADDITIONAL]').join('//[ADDITIONAL]');

    return filecontent;
}

function handleFileSelect(K3D, evt) {
    const { files } = evt.dataTransfer;
    const HTMLSnapshotReader = new FileReader();
    const BinarySnapshotReader = new FileReader();
    const STLReader = new FileReader();

    evt.stopPropagation();
    evt.preventDefault();

    HTMLSnapshotReader.onload = function (event) {
        const snapshot = K3D.extractSnapshot(event.target.result);

        if (snapshot[1]) {
            K3D.setSnapshot(snapshot[1]);
        }
    };

    BinarySnapshotReader.onload = function (event) {
        K3D.setSnapshot(fflate.unzlibSync(event.target.result));
    };

    STLReader.onload = function (event) {
        const stl = event.target.result;

        K3D.load({
            objects: [{
                type: 'STL',
                model_matrix: {
                    data: [
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1],
                },
                binary: stl,
                visible: true,
                color: 0x0000ff,
                wireframe: false,
                flat_shading: false,
            }],
        });
    };

    if (files.length > 0) {
        if (files[0].name.substr(-4).toLowerCase() === 'html') {
            HTMLSnapshotReader.readAsText(files[0]);
            return;
        }

        if (files[0].name.substr(-3).toLowerCase() === 'stl') {
            STLReader.readAsArrayBuffer(files[0]);
            return;
        }

        BinarySnapshotReader.readAsArrayBuffer(files[0]);
    }
}

function handleDragOver(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
}

function snapshotGUI(gui, K3D) {
    const obj = {
        snapshot() {
            let data = getHTMLSnapshot(K3D, 9);
            let filename = `K3D-snapshot-${Date.now()}.html`;

            if (K3D.parameters.name) {
                filename = `${K3D.parameters.name}.html`;
            }

            data = new Blob([data], { type: 'text/plain;charset=utf-8' });
            FileSaver.saveAs(data, filename);
        },
    };
    gui.add(obj, 'snapshot').name('Snapshot HTML');

    // Setup the dnd listeners.
    const targetDomNode = K3D.getWorld().targetDOMNode;

    targetDomNode.addEventListener('dragover', handleDragOver, false);
    targetDomNode.addEventListener('drop', handleFileSelect.bind(null, K3D), false);
}

module.exports = {
    snapshotGUI,
    getHTMLSnapshot,
};
