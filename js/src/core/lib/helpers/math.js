module.exports = {
    pow10ceil(x) {
        return Math.pow(10, Math.ceil(Math.log10(x)));
    },
    fmod(a, b) {
        return Number((a - (Math.floor(a / b) * b)).toPrecision(8));
    },
    decodeFloat16(binary) {
        var exponent = (binary & 0x7C00) >> 10,
            fraction = binary & 0x03FF;
        return (binary >> 15 ? -1 : 1) * (
            exponent ?
                (
                    exponent === 0x1F ?
                        fraction ? NaN : Infinity :
                        Math.pow(2, exponent - 15) * (1 + fraction / 0x400)
                ) :
                6.103515625e-5 * (fraction / 0x400)
        );
    }
};
