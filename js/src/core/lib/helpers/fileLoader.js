/**
 * Fetch a URL and hand the response to `callback`.
 *
 * `onError` is optional but real: callers (snapshot.js) always passed a third argument that
 * could never fire, because this used to invoke `callback` for every finished request no
 * matter the HTTP status - so a 404 was delivered as if it were the file.
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
