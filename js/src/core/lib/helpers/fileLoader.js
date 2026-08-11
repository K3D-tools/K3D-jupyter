/**
 * Fetch a URL and hand the response to `callback`.
 *
 * `callback` runs only for a successful response; an error status goes to `onError`, so a 404
 * body is never mistaken for the file.
 *
 * @param {String} url
 * @param {Function} callback  called with the response body on success
 * @param {Function} [onError] called with an Error on transport failure or a non-2xx status
 */
module.exports = function (url, callback, onError) {
    const xhrLoad = new XMLHttpRequest();

    function fail(message) {
        if (typeof onError === 'function') {
            onError(new Error(message));
        }
    }

    xhrLoad.open('GET', url, true);

    xhrLoad.onreadystatechange = function () {
        if (xhrLoad.readyState !== 4) {
            return;
        }

        // status is 0 for file:// URLs, where a successful read still reports no status.
        const ok = (xhrLoad.status >= 200 && xhrLoad.status < 300)
            || (xhrLoad.status === 0 && xhrLoad.response);

        if (ok) {
            callback(xhrLoad.response);
        } else {
            fail(`Failed to load ${url} (HTTP ${xhrLoad.status})`);
        }
    };

    xhrLoad.onerror = function () {
        fail(`Network error while loading ${url}`);
    };

    xhrLoad.send(null);
};
