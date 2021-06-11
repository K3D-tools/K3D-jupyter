module.exports = {
    pow10ceil(x) {
        return Math.pow(10, Math.ceil(Math.log10(x)));
    },
    fmod(a, b) {
        return Number((a - (Math.floor(a / b) * b)).toPrecision(8));
    },
};
