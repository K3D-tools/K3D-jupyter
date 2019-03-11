'use strict';

module.exports = {
    pow10ceil: function (x) {
        return Math.pow(10, Math.ceil(Math.log10(x)));
    },
    fmod: function (a, b) {
        return Number((a - (Math.floor(a / b) * b)).toPrecision(8));
    }
};
