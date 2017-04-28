var originalConsoleLog = console.log;

window.TestHelpers = {};

window.TestHelpers.createTestCanvas = function () {

    'use strict';

    var canvasHolder = document.createElement('div');

    canvasHolder.id = 'canvasTarget' + (new Date() * Math.random());
    canvasHolder.style.width = '1024px';
    canvasHolder.style.height = '768px';

    return document.body.appendChild(canvasHolder);
};

window.TestHelpers.removeTestCanvas = function (canvasHolder) {

    'use strict';

    document.body.removeChild(canvasHolder);
};

console.log = function () {

    'use strict';

    if (arguments[0] !== 'THREE.WebGLRenderer' && parseInt(arguments[arguments[1]]) > 0) {
        originalConsoleLog.apply(originalConsoleLog, arguments);
    }
};
