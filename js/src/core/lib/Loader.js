const validateAndPrepareObject = require('./helpers/objectValidator');
const _ = require('../../lodash');
const timeSeries = require('./timeSeries');
const { error } = require('./Error');

/**
 * @method K3D.Loader
 * @description K3D objects objectProvider
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {Object} data K3D JSON with objects
 */
function loader(K3D, data) {
    let objectProvider;
    let startTime;
    const objectsPromieses = [];
    let K3DObjectPromise;

    try {
        data.objects.forEach((json, i) => {
            K3DObjectPromise = false;

            validateAndPrepareObject(K3D, json);

            objectProvider = json && K3D.Provider.Objects[json.type];

            if (typeof (objectProvider) === 'undefined') {
                error('Loader Error', `Unsupported object type ${json.type}`);
                return;
            }

            startTime = new Date().getTime();

            const interpolated = timeSeries.interpolateTimeSeries(json, K3D.parameters.time);
            const changes = (data.changes && data.changes[i]) || interpolated.changes || {};

            if (objectProvider.update && !_.isEmpty(changes)) {
                const obj = K3D.getObjectById(interpolated.json.id);

                if (typeof (obj) !== 'undefined') {
                    K3DObjectPromise = objectProvider.update(json, changes, obj, K3D);
                }
            }

            if (!K3DObjectPromise) {
                Object.keys(changes).forEach((key) => {
                    if (changes[key] === null || !changes[key].timeSeries) {
                        interpolated.json[key] = changes[key];
                    }
                });

                K3DObjectPromise = objectProvider.create(interpolated.json, K3D)
                    .then((K3DObject) => {
                        K3DObject.visible = interpolated.json.visible;

                        const objectNumber = K3D.addOrUpdateObject(json, K3DObject);

                        K3DObject.K3DIdentifier = json.id || (`K3DAutoIncrement_${objectNumber}`);
                        json.id = K3DObject.K3DIdentifier;

                        console.log(
                            `K3D: Object type "${json.type}" loaded in: `,
                            (new Date().getTime() - startTime) / 1000,
                            's',
                        );

                        return { json, obj: K3DObject };
                    })
                    .catch((err) => {
                        console.error(err);
                        error('Loader Error', `Object of type "${json.type}" was not loaded.`);
                    });
            }

            objectsPromieses.push(K3DObjectPromise);
        });

        return Promise.all(objectsPromieses);
    } catch (e) {
        error('Loader Error', 'K3D Loader failed, please consult browser error console!', false);
        throw e;
    }
}

module.exports = loader;
