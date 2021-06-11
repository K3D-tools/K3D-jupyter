module.exports = function (url, callback) {
    const xhrLoad = new XMLHttpRequest();

    xhrLoad.open('GET', url, true);

    xhrLoad.onreadystatechange = function () {
        if (xhrLoad.readyState === 4) {
            callback(xhrLoad.response);
        }
    };

    xhrLoad.send(null);
};
