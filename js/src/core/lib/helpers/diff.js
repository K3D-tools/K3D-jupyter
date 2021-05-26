module.exports = {
    diff(config, prevConfig) {
        const diff = {};

        Object.keys(config).forEach((key) => {
            if (config[key] !== prevConfig[key]) {
                diff[key] = true;
            }
        });

        return diff;
    },
};
