const THREE = require('three');
const FileSaver = require('file-saver');
const { GLTFExporter } = require('three/examples/jsm/exporters/GLTFExporter.js');

// Text is drawn as DOM over the canvas and TextureText as camera-facing sprites, so neither has
// geometry to hand over. Label is listed because it also keeps a stub line for its leader, which
// would otherwise export as a stray segment with no text on it. Text and Text2d never reach
// K3DObjects at all, so they need no entry here.
const DOM_TYPES = ['Label', 'TextureText'];

/**
 * glTF carries triangles and PBR materials, nothing else. Whatever K3D draws with a custom shader
 * has no surface to hand over - a volume is a unit cube ray-marched in the fragment stage, a
 * billboarded point is one vertex, a thick line is a ribbon the vertex stage widens - so exporting
 * their raw buffers would ship a solid that looks nothing like the render. Skipping them loses
 * those objects; exporting them would lie about them.
 * @param {THREE.Object3D} node
 * @return {boolean}
 */
function isExportable(node) {
    if (!(node.isMesh || node.isLine || node.isPoints)) {
        return false;
    }

    const material = Array.isArray(node.material) ? node.material[0] : node.material;

    if (!material || material.isShaderMaterial) {
        return false;
    }

    const position = node.geometry && node.geometry.attributes.position;

    return !!position && position.count > 0;
}

/**
 * glTF colours are linear, K3D's are displayed values: converts them for the export.
 * @param {Array<THREE.Object3D>} nodes the nodes being exported
 * @return {Function} restores every colour it converted
 */
function linearizeColors(nodes) {
    const materials = new Map();
    const attributes = [];

    nodes.forEach((node) => {
        const material = Array.isArray(node.material) ? node.material[0] : node.material;

        if (material && !materials.has(material)) {
            materials.set(material, {
                color: material.color ? material.color.clone() : null,
                emissive: material.emissive ? material.emissive.clone() : null,
            });

            if (material.color) {
                material.color.convertSRGBToLinear();
            }
            if (material.emissive) {
                material.emissive.convertSRGBToLinear();
            }
        }

        const color = node.geometry && node.geometry.getAttribute('color');

        if (color && !attributes.some(([geometry]) => geometry === node.geometry)) {
            const linear = color.clone();
            const converted = new THREE.Color();

            for (let i = 0; i < linear.count; i++) {
                converted.setRGB(linear.getX(i), linear.getY(i), linear.getZ(i)).convertSRGBToLinear();
                linear.setXYZ(i, converted.r, converted.g, converted.b);
            }

            attributes.push([node.geometry, color]);
            node.geometry.setAttribute('color', linear);
        }
    });

    return () => {
        materials.forEach((saved, material) => {
            if (saved.color) {
                material.color.copy(saved.color);
            }
            if (saved.emissive) {
                material.emissive.copy(saved.emissive);
            }
        });
        attributes.forEach(([geometry, color]) => {
            geometry.setAttribute('color', color);
        });
    };
}

/**
 * Export the scene geometry as a binary glTF (.glb).
 * @memberof K3D.Core
 * @param {Object} K3D
 * @return {Promise<ArrayBuffer>}
 */
function getGLTF(K3D) {
    const world = K3D.getWorld();
    const hidden = [];
    const renamed = [];
    const skipped = [];

    const hide = (node) => {
        if (node.visible) {
            hidden.push(node);
            node.visible = false;
        }
    };

    Object.keys(world.ObjectsById).forEach((id) => {
        const object = world.ObjectsById[id];
        const json = world.ObjectsListJson[id];

        if (!object || !json) {
            return;
        }

        const label = json.name || `${json.type}_${id}`;
        let exportable = false;

        if (DOM_TYPES.indexOf(json.type) !== -1) {
            hide(object);
        } else {
            object.traverse((node) => {
                exportable = exportable || isExportable(node);
            });
        }

        if (!exportable) {
            // an object the user hid still counts as exportable, so this reports only the ones
            // the format itself cannot take - otherwise an empty file has no explanation
            skipped.push(label);

            return;
        }

        // without this every node lands in the file unnamed and the importer numbers them
        renamed.push([object, object.name]);
        object.name = label;
    });

    world.K3DObjects.traverse((node) => {
        if ((node.isMesh || node.isLine || node.isPoints) && !isExportable(node)) {
            hide(node);
        }
    });

    if (skipped.length > 0) {
        console.warn(
            `K3D: glTF holds no geometry for ${skipped.length} object(s), left out of the export: ${
                skipped.join(', ')}`,
        );
    }

    const exported = [];

    world.K3DObjects.traverse((node) => {
        if (node.visible && isExportable(node)) {
            exported.push(node);
        }
    });

    const restoreColors = linearizeColors(exported);

    const restore = () => {
        restoreColors();
        hidden.forEach((node) => {
            node.visible = true;
        });
        renamed.forEach(([object, name]) => {
            object.name = name;
        });
    };

    return new Promise((resolve, reject) => {
        new GLTFExporter().parse(
            world.K3DObjects,
            (result) => {
                restore();
                resolve(result);
            },
            (error) => {
                restore();
                reject(error);
            },
            { binary: true, onlyVisible: true },
        );
    });
}

function gltfGUI(gui, K3D) {
    const obj = {
        gltf() {
            getGLTF(K3D).then((glb) => {
                const name = K3D.parameters.name || `K3D-${Date.now()}`;

                FileSaver.saveAs(new Blob([glb], { type: 'model/gltf-binary' }), `${name}.glb`);
            }, (e) => {
                console.error('Failed to export glTF.', e);
            });
        },
    };

    gui.add(obj, 'gltf').name('Export glTF');
}

module.exports = {
    gltfGUI,
    getGLTF,
};
