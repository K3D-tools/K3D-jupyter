'use strict';

var domHelper,
    defaultTimeout = 5000;

function setupErrorDomHelper() {
    domHelper = document.createElement('ul');
    domHelper.id = 'K3D-error-console';
    domHelper.style.cssText = [
        'font: 12px sans-serif',
        'color: #FFFFFF',
        'position: absolute',
        'top: 0px',
        'right: 0px',
        'padding: 0px',
        'margin: 0px',
        'list-style: none outside none'
    ].join(';');

    document.body.appendChild(domHelper);
}

function setupSingleErrorDomHelper() {

    var domHelperItem;

    domHelperItem = document.createElement('li');

    domHelperItem.style.cssText = [
        'background-color: #B50F0F',
        'padding: 5px',
        'margin-bottom: 1px'
    ].join(';');

    domHelper.appendChild(domHelperItem);

    return domHelperItem;
}

function error(title, message, permanent) {

    var domHelperItem;

    if (!(domHelper instanceof Node)) {
        setupErrorDomHelper();
    }

    domHelperItem = setupSingleErrorDomHelper();

    domHelperItem.innerHTML = [
        '<b>',
        title || 'Error',
        '</b>: ',
        message
    ].join('');

    if (!permanent) {
        setTimeout(function () {
            domHelper.removeChild(domHelperItem);
        }, defaultTimeout);
    }
}

if (window.Detector && !window.Detector.webgl) {
    error('General Error',
        'It seams that your browser has no support for WebGL, while it\'s required. Please use a different one.');
}

module.exports = {
    error: error,
    defaultTimeout: defaultTimeout
};


