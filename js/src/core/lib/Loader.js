'use strict';

var validateAndPrepareObject = require('./helpers/objectValidator'),
    timeSeries = require('./timeSeries'),
    error = require('./Error').error;

/**
 * @method K3D.Loader
 * @description K3D objects objectProvider
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {Object} json K3D JSON with objects
 */
function loader(K3D, json) {

    var objectProvider, startTime, objectsPromieses = [], K3DObjectPromise;

    try {
        json.objects.forEach(function (json) {
            K3DObjectPromise = false;

            validateAndPrepareObject(K3D, json);

            objectProvider = json && K3D.Provider.Objects[json.type];

            if (typeof (objectProvider) === 'undefined') {
                error('Loader Error', 'Unsupported object type ' + json.type);
                return;
            }

            startTime = new Date().getTime();

            var interpolated_object = timeSeries.interpolateTimeSeries(json, K3D.parameters.time);

            if (objectProvider.update) {
                var obj = K3D.getObjectById(interpolated_object.id), prevConfig;

                if (typeof(obj) !== 'undefined') {
                    prevConfig = K3D.getWorld().ObjectsListJson[interpolated_object.id];
                    K3DObjectPromise = objectProvider.update(interpolated_object, prevConfig, obj, K3D);

                    if (K3DObjectPromise) {
                        console.log('K3D: Object type "' + json.type + '" updated in: ',
                            (new Date().getTime() - startTime) / 1000, 's');
                    }
                }
            }

            if (!K3DObjectPromise) {
                K3DObjectPromise = objectProvider.create(interpolated_object, K3D)
                    .then(function (K3DObject) {
                        var objectNumber;

                        objectNumber = K3D.addOrUpdateObject(json, K3DObject);

                        K3DObject.K3DIdentifier = json.id || ('K3DAutoIncrement_' + objectNumber);
                        json.id = K3DObject.K3DIdentifier;

                        console.log('K3D: Object type "' + json.type + '" loaded in: ',
                            (new Date().getTime() - startTime) / 1000, 's');

                        return {json: json, obj: K3DObject};
                    })
                    .catch(function () {
                        error('Loader Error', 'Object of type "' + json.type + '" was not loaded.');
                    });
            }

            objectsPromieses.push(K3DObjectPromise);
        });

        return Promise.all(objectsPromieses);
    } catch (e) {
        error('Loader Error', 'K3D Loader failed, please consult browser error console!', true);
        throw e;
    }
}

module.exports = loader;
