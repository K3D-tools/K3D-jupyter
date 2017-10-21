'use strict';

var screenfull = require('screenfull');

function fullscreen(container, toolbarContainer, currentWindow) {
    var style = {
            initial: container.style.cssText,
            fullscreen: [
                ';height: auto',
                ';width: auto',
                ';position: fixed',
                ';bottom: 0',
                ';left: 0',
                ';right: 0',
                ';top: 0'
            ].join('')
        },
        element = document.createElement('img'),
        fullscreenChangeListener = function () {
            container.style.cssText = style.initial + (screenfull.isFullscreen ? style.fullscreen : '');
            element.style.display = screenfull.isFullscreen ? 'none' : 'initial';
            currentWindow.dispatchEvent(new Event('resize'));
        };

    element.style.cssText = [
        'background: white',
        'cursor: pointer',
        'width: 24px',
        'height: 24px',
        'margin: 0 3px',
        'padding: 2px'
    ].join(';');

    /* jshint -W101 */
    element.src =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABXElEQVRIS7WVbU2EQQyE5xwgAQeAAg4FSAAUAEo4HIADHIADkIAEJJBn8zZpeu12+cEml0vebWf6Me3u9M9nt+GfSLqRxH88P5Kek+8zn29Jr/gYwV7S+ySZK0kf4f5B0lPns0JAJLdFBpCeFSQjqI6gAjdMylSRLBG8SLqblAECSnue2CwR4FeRzMDxKwlG9zdVWWCRJAN/3HplPUkJfM0BRbqRJAOnjNj7nlxI+rImA4JUoxQzEurta27gvvGngPs5mM1zJPG2EfwIx2fwV5IWfDWDSi2dhEfAXQadFFsSv+zut6XGcuNUakEMmbqsxAjg0hZkXBV0Hv1y4oT6mlcSBhw/gisHbcirkSL3keRtk7qt/OVVMVPLTMJLBIz/YaZfSUR+ndgsEVhPrPERx9c83i0R4FSRsA4+i2cWv2UCjMfiCiF2T+bRsiPd7NHnAeeXndanm+Smv/31L00TYRmfNUExAAAAAElFTkSuQmCC';
    /* jshint +W101 */

    element.setAttribute('title', 'Fullscreen');

    currentWindow.addEventListener(screenfull.raw.fullscreenchange, fullscreenChangeListener);
    element.addEventListener('click', function () {
        screenfull.request(container);
    });

    toolbarContainer.insertBefore(element, toolbarContainer.firstChild);
}

module.exports = {
    isAvailable: function () {
        return screenfull.enabled;
    },

    initialize: fullscreen
};
