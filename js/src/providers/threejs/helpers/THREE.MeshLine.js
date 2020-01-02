'use strict';
//jshint ignore: start
//jscs:disable

module.exports = function (THREE) {
    function MeshLine() {
        this.positions = [];
        this.previous = [];
        this.next = [];
        this.side = [];
        this.width = [];
        this.indices_array = [];
        this.uvs = [];
        this.counters = [];
        this.geometry = new THREE.BufferGeometry();
    }

    MeshLine.prototype.setGeometry = function (g, segments, widths, colors, uvs) {
        var j, c, l;

        if (g instanceof Float32Array || g instanceof Array) {
            this.positions = new Float32Array(g.length * 2);
            this.counters = new Float32Array(2 * g.length / 3);

            for (j = 0; j < g.length; j += 3) {
                c = j / g.length;

                this.positions[j * 2] = this.positions[j * 2 + 3] = g[j];
                this.positions[j * 2 + 1] = this.positions[j * 2 + 4] = g[j + 1];
                this.positions[j * 2 + 2] = this.positions[j * 2 + 5] = g[j + 2];

                this.counters[j * 2] = this.counters[j * 2 + 1] = c;
            }
        }

        l = this.positions.length / 6;

        this.width = new Float32Array(l * 2);
        this.colors = new Float32Array(l * 6);
        this.uvs = new Float32Array(l * 4);

        for (j = 0; j < l; j++) {
            if (widths) {
                this.width[j * 2] = this.width[j * 2 + 1] = widths[j];
            } else {
                this.width[j * 2] = this.width[j * 2 + 1] = 1.0;
            }

            if (colors) {
                this.colors[j * 6] = this.colors[j * 6 + 3] = colors[j * 3];
                this.colors[j * 6 + 1] = this.colors[j * 6 + 4] = colors[j * 3 + 1];
                this.colors[j * 6 + 2] = this.colors[j * 6 + 5] = colors[j * 3 + 2];
            } else {
                this.colors[j * 6] = this.colors[j * 6 + 1] = this.colors[j * 6 + 2] =
                    this.colors[j * 6 + 3] = this.colors[j * 6 + 4] = this.colors[j * 6 + 5] = 1.0;
            }

            if (uvs) {
                this.uvs[j * 4] = this.uvs[j * 4 + 2] = uvs[j];
                this.uvs[j * 4 + 1] = 0;
                this.uvs[j * 4 + 3] = 1;
            } else {
                this.uvs[j * 4] = this.uvs[j * 4 + 2] = j / (l - 1);
                this.uvs[j * 4 + 1] = 0;
                this.uvs[j * 4 + 3] = 1;
            }
        }

        this.process(segments);
    };

    MeshLine.prototype.compareV3 = function (a, b) {
        var aa = a * 6,
            ab = b * 6;

        return (this.positions[aa] === this.positions[ab]) && (this.positions[aa + 1] === this.positions[ab + 1]) && (this.positions[aa + 2] === this.positions[ab + 2]);

    };

    MeshLine.prototype.copyV3 = function (a) {

        var aa = a * 6;
        return [this.positions[aa], this.positions[aa + 1], this.positions[aa + 2]];

    };

    MeshLine.prototype.process = function (segments) {
        var l = this.positions.length / 6, j, n, k, v, o;

        this.previous = new Float32Array(12 * l / 2);
        this.next = new Float32Array(12 * l / 2);
        this.side = new Float32Array(l * 2);
        this.indices_array = new Uint32Array((Math.max(l, 1) - 1) * 6);

        for (j = 0; j < l; j++) {
            this.side[j * 2] = 1;
            this.side[j * 2 + 1] = -1;
        }

        if (segments) {
            for (j = 0; j < l; j += 2) {
                o = j / 2 * 12;

                this.previous[o] = this.previous[o + 3] = this.previous[o + 6] = this.previous[o + 9] = this.positions[j * 6];
                this.previous[o + 1] = this.previous[o + 4] = this.previous[o + 7] = this.previous[o + 10] = this.positions[j * 6 + 1];
                this.previous[o + 2] = this.previous[o + 5] = this.previous[o + 8] = this.previous[o + 11] = this.positions[j * 6 + 2];
            }

            for (j = 0; j < l; j += 2) {
                k = j + 1;
                o = j / 2 * 12;

                this.next[o] = this.next[o + 3] = this.next[o + 6] = this.next[o + 9] = this.positions[k * 6];
                this.next[o + 1] = this.next[o + 4] = this.next[o + 7] = this.next[o + 10] = this.positions[k * 6 + 1];
                this.next[o + 2] = this.next[o + 5] = this.next[o + 8] = this.next[o + 11] = this.positions[k * 6 + 2];
            }

            for (j = 0; j < l - 1; j += 2) {
                n = j * 2;
                this.indices_array[j * 6] = n;
                this.indices_array[j * 6 + 1] = n + 1;
                this.indices_array[j * 6 + 2] = n + 2;
                this.indices_array[j * 6 + 3] = n + 2;
                this.indices_array[j * 6 + 4] = n + 1;
                this.indices_array[j * 6 + 5] = n + 3;
            }
        } else {
            if (this.compareV3(0, l - 1)) {
                v = this.copyV3(l - 2);
            } else {
                v = this.copyV3(0);
            }

            this.previous[0] = this.previous[3] = v[0];
            this.previous[1] = this.previous[4] = v[1];
            this.previous[2] = this.previous[5] = v[2];

            for (j = 0; j < l - 1; j++) {
                v = this.copyV3(j);

                this.previous[6 + j * 6] = this.previous[6 + j * 6 + 3] = v[0];
                this.previous[6 + j * 6 + 1] = this.previous[6 + j * 6 + 4] = v[1];
                this.previous[6 + j * 6 + 2] = this.previous[6 + j * 6 + 5] = v[2];
            }

            for (j = 1; j < l; j++) {
                v = this.copyV3(j);

                this.next[(j - 1) * 6] = this.next[(j - 1) * 6 + 3] = v[0];
                this.next[(j - 1) * 6 + 1] = this.next[(j - 1) * 6 + 4] = v[1];
                this.next[(j - 1) * 6 + 2] = this.next[(j - 1) * 6 + 5] = v[2];
            }

            if (this.compareV3(l - 1, 0)) {
                v = this.copyV3(1);
            } else {
                v = this.copyV3(l - 1);
            }

            this.next[(l - 1) * 6] = this.next[(l - 1) * 6 + 3] = v[0];
            this.next[(l - 1) * 6 + 1] = this.next[(l - 1) * 6 + 4] = v[1];
            this.next[(l - 1) * 6 + 2] = this.next[(l - 1) * 6 + 5] = v[2];

            for (j = 0; j < l - 1; j++) {
                n = j * 2;

                this.indices_array[j * 6] = n;
                this.indices_array[j * 6 + 1] = n + 1;
                this.indices_array[j * 6 + 2] = n + 2;
                this.indices_array[j * 6 + 3] = n + 2;
                this.indices_array[j * 6 + 4] = n + 1;
                this.indices_array[j * 6 + 5] = n + 3;
            }
        }

        if (!this.attributes) {
            this.attributes = {
                position: new THREE.BufferAttribute(this.positions, 3),
                previous: new THREE.BufferAttribute(this.previous, 3),
                next: new THREE.BufferAttribute(this.next, 3),
                side: new THREE.BufferAttribute(this.side, 1),
                width: new THREE.BufferAttribute(this.width, 1),
                uv: new THREE.BufferAttribute(this.uvs, 2),
                index: new THREE.BufferAttribute(this.indices_array, 1),
                counters: new THREE.BufferAttribute(this.counters, 1),
                colors: new THREE.BufferAttribute(this.colors, 3)
            };
        } else {
            this.attributes.position.copyArray(this.positions);
            this.attributes.position.needsUpdate = true;
            this.attributes.previous.copyArray(this.previous);
            this.attributes.previous.needsUpdate = true;
            this.attributes.next.copyArray(this.next);
            this.attributes.next.needsUpdate = true;
            this.attributes.side.copyArray(this.side);
            this.attributes.side.needsUpdate = true;
            this.attributes.width.copyArray(this.width);
            this.attributes.width.needsUpdate = true;
            this.attributes.uv.copyArray(this.uvs);
            this.attributes.uv.needsUpdate = true;
            this.attributes.index.copyArray(this.indices_array);
            this.attributes.index.needsUpdate = true;
            this.attributes.colors.copyArray(this.colors);
            this.attributes.colors.needsUpdate = true;
        }

        this.geometry.setAttribute('position', this.attributes.position);
        this.geometry.setAttribute('previous', this.attributes.previous);
        this.geometry.setAttribute('next', this.attributes.next);
        this.geometry.setAttribute('side', this.attributes.side);
        this.geometry.setAttribute('width', this.attributes.width);
        this.geometry.setAttribute('uv', this.attributes.uv);
        this.geometry.setAttribute('counters', this.attributes.counters);
        this.geometry.setAttribute('colors', this.attributes.colors);

        this.geometry.setIndex(this.attributes.index);
    };

    function MeshLineMaterial(parameters) {

        var vertexShaderSource = [
                'precision highp float;',
                '',
                'attribute vec3 previous;',
                'attribute vec3 next;',
                'attribute vec3 colors;',
                'attribute float side;',
                'attribute float width;',
                'attribute float counters;',
                '',
                'uniform vec2 resolution;',
                'uniform float lineWidth;',
                'uniform vec3 color;',
                'uniform float opacity;',
                'uniform float near;',
                'uniform float far;',
                'uniform float sizeAttenuation;',
                '',
                'varying vec2 vUV;',
                'varying vec4 vColor;',
                'varying float vCounters;',
                '',
                '#include <clipping_planes_pars_vertex>',
                '',
                'vec2 fix( vec4 i, float aspect ) {',
                '',
                '    vec2 res = i.xy / i.w;',
                '    res.x *= aspect;',
                '    vCounters = counters;',
                '    return res;',
                '',
                '}',
                '',
                'void main() {',
                '',
                '    float aspect = resolution.x / resolution.y;',
                '    float pixelWidthRatio = 1. / (resolution.x * projectionMatrix[0][0]);',
                '',
                '    vColor = vec4( color * colors, opacity );',
                '    vUV = uv;',
                '',
                '    #if NUM_CLIPPING_PLANES > 0 && ! defined( PHYSICAL ) && ! defined( PHONG )',
                '    vViewPosition = -(modelViewMatrix * vec4(position, 1.0)).xyz;',
                '    #endif',
                '',
                '    mat4 m = projectionMatrix * modelViewMatrix;',
                '    vec4 finalPosition = m * vec4( position, 1.0 );',
                '    vec4 prevPos = m * vec4( previous, 1.0 );',
                '    vec4 nextPos = m * vec4( next, 1.0 );',
                '',
                '    vec2 currentP = fix( finalPosition, aspect );',
                '    vec2 prevP = fix( prevPos, aspect );',
                '    vec2 nextP = fix( nextPos, aspect );',
                '',
                '    float pixelWidth = finalPosition.w * pixelWidthRatio;',
                '    float w = 1.8 * pixelWidth * lineWidth * width;',
                '',
                '    if( sizeAttenuation == 1. ) {',
                '        w = max(1.8 * pixelWidth * 2.0, 1.8 * lineWidth * width);',
                '    }',
                '',
                '    vec2 dir;',
                '    if( nextP == currentP ) dir = normalize( currentP - prevP );',
                '    else if( prevP == currentP ) dir = normalize( nextP - currentP );',
                '    else {',
                '        vec2 dir1 = normalize( currentP - prevP );',
                '        vec2 dir2 = normalize( nextP - currentP );',
                '        dir = normalize( dir1 + dir2 );',
                '',
                '        vec2 perp = vec2( -dir1.y, dir1.x );',
                '        vec2 miter = vec2( -dir.y, dir.x );',
                '        //w = clamp( w / dot( miter, perp ), 0., 4. * lineWidth * width );',
                '',
                '    }',
                '',
                '    //vec2 normal = ( cross( vec3( dir, 0. ), vec3( 0., 0., 1. ) ) ).xy;',
                '    vec2 normal = vec2( -dir.y, dir.x );',
                '    normal.x /= aspect;',
                '    normal *= .5 * w;',
                '',
                '    vec4 offset = vec4( normal * side, 0.0, 1.0 );',
                '    finalPosition.xy += offset.xy;',
                '',
                '    gl_Position = finalPosition;',
                '',
                '}'],
            fragmentShaderSource = [
                'precision mediump float;',
                '',
                'uniform sampler2D map;',
                'uniform float useMap;',
                'uniform float visibility;',
                '',
                'varying vec2 vUV;',
                'varying vec4 vColor;',
                'varying float vCounters;',
                '',
                '#include <clipping_planes_pars_fragment>',
                '',
                'void main() {',
                '',
                ' #include <clipping_planes_fragment>',
                '',
                '    vec4 c = vColor;',
                '    if( useMap == 1. ) c *= texture2D( map, vUV);',
                '    gl_FragColor = c;',
                '    gl_FragColor.a *= step(vCounters, visibility);',
                '}'],
            material;

        function check(v, d) {
            if (v === undefined) {
                return d;
            }

            return v;
        }

        THREE.Material.call(this);

        parameters = parameters || {};

        this.lineWidth = check(parameters.lineWidth, 1);
        this.map = check(parameters.map, null);
        this.useMap = check(parameters.useMap, 0);
        this.color = check(parameters.color, new THREE.Color(0xffffff));
        this.opacity = check(parameters.opacity, 1);
        this.resolution = check(parameters.resolution, new THREE.Vector2(1, 1));
        this.sizeAttenuation = check(parameters.sizeAttenuation, 1);
        this.near = check(parameters.near, 1);
        this.far = check(parameters.far, 1);
        this.visibility = check(parameters.visibility, 1);

        material = new THREE.ShaderMaterial({
            uniforms: {
                lineWidth: {type: 'f', value: this.lineWidth},
                map: {type: 't', value: this.map},
                useMap: {type: 'f', value: this.useMap},
                color: {type: 'c', value: this.color},
                opacity: {type: 'f', value: this.opacity},
                resolution: {type: 'v2', value: this.resolution},
                sizeAttenuation: {type: 'f', value: this.sizeAttenuation},
                near: {type: 'f', value: this.near},
                far: {type: 'f', value: this.far},
                visibility: {type: 'f', value: this.visibility}
            },
            vertexShader: vertexShaderSource.join('\r\n'),
            fragmentShader: fragmentShaderSource.join('\r\n'),
            clipping: true
        });

        delete parameters.lineWidth;
        delete parameters.map;
        delete parameters.useMap;
        delete parameters.color;
        delete parameters.opacity;
        delete parameters.resolution;
        delete parameters.sizeAttenuation;
        delete parameters.near;
        delete parameters.far;
        delete parameters.visibility;

        material.type = 'MeshLineMaterial';
        material.setValues(parameters);

        return material;
    }

    MeshLineMaterial.prototype = Object.create(THREE.Material.prototype);
    MeshLineMaterial.prototype.constructor = MeshLineMaterial;

    MeshLineMaterial.prototype.copy = function (source) {
        THREE.Material.prototype.copy.call(this, source);

        this.lineWidth = source.lineWidth;
        this.map = source.map;
        this.useMap = source.useMap;
        this.color.copy(source.color);
        this.opacity = source.opacity;
        this.resolution.copy(source.resolution);
        this.sizeAttenuation = source.sizeAttenuation;
        this.near = source.near;
        this.far = source.far;
        this.visibility = source.visibility;

        return this;
    };

    return {
        MeshLine: MeshLine,
        MeshLineMaterial: MeshLineMaterial
    };
};
