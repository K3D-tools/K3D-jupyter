'use strict';

var validateAndPrepareObject = require('./helpers/objectValidator'),
    error = require('./Error').error;

/**
 * @method K3D.Loader
 * @description K3D objects loader
 * @memberof K3D
 * @param {K3D.Core} K3D A K3D instance to load objects into
 * @param {Object} json K3D JSON with objects
 */
function loader(K3D, json) {

    var loader, startTime, objectsPromieses = [], K3DObjectPromise;

    try {
        json.objects.forEach(function (object) {
            validateAndPrepareObject(K3D, object);

            loader = object && K3D.Provider.Objects[object.type];

            if (typeof (loader) === 'undefined') {
                error('Loader Error', 'Unsupported object type ' + object.type);
                return;
            }

            startTime = new Date().getTime();

            K3DObjectPromise = loader(object, K3D)
                .then(function (K3DObject) {
                    var objectNumber;

                    objectNumber = K3D.addObject(K3DObject);

                    K3DObject.K3DIdentifier = object.id || ('K3DAutoIncrement_' + objectNumber);
                    object.id = K3DObject.K3DIdentifier;

                    console.log('K3D: Object type "' + object.type + '" loaded in: ',
                        (new Date().getTime() - startTime) / 1000, 's');

                    return object;
                })
                .catch(function () {
                    error('Loader Error', 'Object of type "' + object.type + '" was not loaded.');
                });

            objectsPromieses.push(K3DObjectPromise);
        });

        return Promise.all(objectsPromieses).then(function (objects) {
            K3D.getWorld().setCameraToFitScene();

            // rebuild scene + re-render
            Promise.all(K3D.rebuild()).then(K3D.render);

            return objects;
        });
    } catch (e) {
        error('Loader Error', 'K3D Loader failed, please consult browser error console!', true);
        throw e;
    }
}

module.exports = loader;
