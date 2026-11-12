import { expect } from 'chai';
import { restore, spy } from 'sinon';

import { Mat4 } from '../../../src/core/math/mat4.js';
import { XrView } from '../../../src/framework/xr/xr-view.js';
import { Camera } from '../../../src/scene/camera.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('XrView', function () {

    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
    });

    afterEach(function () {
        restore();
        app.destroy();
        app = null;
        jsdomTeardown();
    });

    describe('#destroy', function () {

        it('stops listening for the loss of the graphics device', function () {
            const onDeviceLost = spy(XrView.prototype, '_onDeviceLost');

            // a view with a camera image texture listens for the loss of the graphics device
            const manager = {
                app,
                views: {
                    supportedColor: true,
                    availableColor: true,
                    supportedDepth: false
                }
            };
            const view = new XrView(manager, { eye: 'none', camera: { width: 4, height: 4 } }, 1);

            app.graphicsDevice.fire('devicelost');
            expect(onDeviceLost.callCount).to.equal(1);

            view.destroy();
            app.graphicsDevice.fire('devicelost');
            expect(onDeviceLost.callCount).to.equal(1);
        });

    });

    describe('#update', function () {

        // a projection as WebXR provides it on WebGPU, with the 0..1 depth range
        const projection = Camera.applyShaderProjectionTransform(new Mat4().setPerspective(90, 1, 0.1, 100), new Mat4(), false, true);

        const updateView = () => {
            const manager = {
                app,
                views: {
                    supportedColor: false,
                    availableColor: false,
                    supportedDepth: false,
                    availableDepth: false
                },
                xrBridge: {
                    getViewport: () => ({ x: 0, y: 0, width: 4, height: 4 })
                }
            };
            const identity = new Mat4().data;
            const xrView = {
                eye: 'none',
                projectionMatrix: new Float32Array(projection.data),
                transform: { matrix: identity, inverse: { matrix: identity } }
            };
            const view = new XrView(manager, xrView, 1);
            view.update({}, xrView);
            return view;
        };

        it('uses the projection of WebXR as provided', function () {
            const view = updateView();
            expect(Array.from(view.projMat.data)).to.deep.equal(Array.from(new Float32Array(projection.data)));
            view.destroy();
        });

        it('reverses the depth range of the projection with reverse-z', function () {
            app.graphicsDevice.isReverseZ = true;
            const view = updateView();
            const expected = Camera.applyReverseZ(new Mat4().set(Array.from(new Float32Array(projection.data))));
            for (let i = 0; i < 16; i++) {
                expect(view.projMat.data[i]).to.be.closeTo(expected.data[i], 1e-6);
            }
            view.destroy();
        });

    });

});
