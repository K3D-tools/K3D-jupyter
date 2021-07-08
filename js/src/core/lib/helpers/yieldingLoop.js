module.exports = function (count, chunksize, callback, finished) {
    let i = 0;
    (function chunk() {
        const end = Math.min(i + chunksize, count);
        for (; i < end; ++i) {
            callback.call(null, i);
        }
        if (i < count) {
            setTimeout(chunk, 0);
        } else if (typeof (finished) !== 'undefined') {
            finished.call(null);
        }
    }());
};
