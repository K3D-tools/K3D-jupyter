let domHelper;
const defaultTimeout = 5000;

function setupErrorDomHelper() {
    domHelper = document.createElement('ul');
    domHelper.id = 'K3D-error-console';
    domHelper.style.cssText = [
        'font: 12px sans-serif',
        'color: #FFFFFF',
        'position: absolute',
        'top: 0px',
        'left: 0px',
        'padding: 0px',
        'z-index: 999',
        'margin: 0px',
        'list-style: none outside none',
    ].join(';');

    document.body.appendChild(domHelper);
}

function setupSingleErrorDomHelper() {
    const domHelperItem = document.createElement('li');

    domHelperItem.style.cssText = [
        'background-color: #B50F0F',
        'padding: 5px',
        'margin-bottom: 1px',
    ].join(';');

    domHelper.appendChild(domHelperItem);

    return domHelperItem;
}

function error(title, message, permanent) {
    if (!(domHelper instanceof Node)) {
        setupErrorDomHelper();
    }

    const domHelperItem = setupSingleErrorDomHelper();

    domHelperItem.innerHTML = [
        '<b>',
        title || 'Error',
        '</b>: ',
        message,
    ].join('');

    if (!permanent) {
        setTimeout(() => {
            domHelper.removeChild(domHelperItem);
        }, defaultTimeout);
    }
}

module.exports = {
    error,
    defaultTimeout,
};
