function Float16Array(array) {
    const d = new Uint16Array(array);

    d.constructor = Float16Array;

    return d;
}

module.exports = Float16Array;
