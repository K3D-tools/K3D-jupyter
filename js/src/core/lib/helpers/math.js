/* eslint-disable no-nested-ternary */

module.exports = {
    pow10ceil(x) {
        return 10 ** Math.ceil(Math.log10(x));
    },
    fmod(a, b) {
        return Number((a - (Math.floor(a / b) * b)).toPrecision(8));
    },
    decodeFloat16(binary) {
        const exponent = (binary & 0x7C00) >> 10;
        const fraction = binary & 0x03FF;
        return (binary >> 15 ? -1 : 1) * (
            exponent
                ? (
                    exponent === 0x1F
                        ? fraction ? NaN : Infinity
                        : 2 ** (exponent - 15) * (1 + fraction / 0x400)
                )
                : 6.103515625e-5 * (fraction / 0x400)
        );
    },
};
