import { expect } from 'chai';

import { Color } from '../../../src/core/math/color.js';
import { Vec2 } from '../../../src/core/math/vec2.js';
import { Vec4 } from '../../../src/core/math/vec4.js';
import { Application } from '../../../src/framework/application.js';
import { Entity } from '../../../src/framework/entity.js';
import { SceneDepthReader } from '../../../src/framework/graphics/scene-depth-reader.js';
import { CULLFACE_NONE, PIXELFORMAT_DEPTH, PIXELFORMAT_RGBA8 } from '../../../src/platform/graphics/constants.js';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import {
    FOG_LINEAR, SHADOW_PCF1_32F, SHADOW_PCF3_16F, SHADOW_PCF5_32F, SHADOW_PCSS_32F, SHADOW_VSM_16F,
    SHADOW_VSM_32F, TONEMAP_NONE
} from '../../../src/scene/constants.js';
import { PlaneGeometry } from '../../../src/scene/geometry/plane-geometry.js';
import { StandardMaterial } from '../../../src/scene/materials/standard-material.js';
import { MeshInstance } from '../../../src/scene/mesh-instance.js';
import { Mesh } from '../../../src/scene/mesh.js';
import { createGraphicsDevice, getTestDeviceType } from '../../device.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// Dawn's null backend validates the API and compiles the shaders, but executes no GPU work, so the
// pixels are only compared on a backend which runs them
const executesGpuWork = !!process.env.PC_WEBGPU_BACKEND && process.env.PC_WEBGPU_BACKEND !== 'null';

const WIDTH = 64;
const HEIGHT = 64;

/**
 * Renders a scene with a standard or a reverse-z device, and returns the pixels.
 *
 * @param {boolean} reverseZ - Whether the device uses reverse-z.
 * @param {(app: Application) => object} build - Builds the scene, and returns the options of the
 * camera rendering it.
 * @param {(device: object) => object} [targetOptions] - Returns additional options of the render
 * target, for the graphics device.
 * @returns {Promise<Uint8Array|null>} The RGBA8 pixels, rows from the top, or null when the backend
 * executes no GPU work.
 */
const renderScene = async (reverseZ, build, targetOptions = () => ({})) => {
    const canvas = document.createElement('canvas');
    const graphicsDevice = createGraphicsDevice(canvas, { reverseZ });
    expect(graphicsDevice.isReverseZ).to.equal(reverseZ);
    const app = new Application(canvas, { graphicsDevice });

    const colorBuffer = new Texture(graphicsDevice, { name: 'Color', width: WIDTH, height: HEIGHT, format: PIXELFORMAT_RGBA8, mipmaps: false });
    const renderTarget = new RenderTarget({ colorBuffer, depth: true, ...targetOptions(graphicsDevice) });

    const cameraOptions = build(app);
    const camera = new Entity('Camera');
    camera.addComponent('camera', {
        clearColor: new Color(0, 0, 0, 1),
        toneMapping: TONEMAP_NONE,
        renderTarget,
        ...cameraOptions.camera
    });
    camera.setPosition(...(cameraOptions.position ?? [0, 0, 0]));
    camera.lookAt(...(cameraOptions.target ?? [0, 0, -1]));
    app.root.addChild(camera);

    // the second frame renders with the shadows the first frame allocated
    app.render();
    app.render();

    let pixels = null;
    if (executesGpuWork) {
        pixels = await colorBuffer.read(0, 0, WIDTH, HEIGHT, { renderTarget, immediate: true });
    }

    renderTarget.destroyTextureBuffers();
    renderTarget.destroy();
    app.destroy();
    return pixels;
};

const createMaterial = (color, lit = true) => {
    const material = new StandardMaterial();
    if (lit) {
        material.diffuse = color;
    } else {
        material.useLighting = false;
        material.diffuse = Color.BLACK;
        material.emissive = color;
    }
    material.update();
    return material;
};

// a box casting a directional shadow onto a ground plane, seen from above at an angle. The shadow
// falls towards the camera, onto the bottom of the view
const shadowScene = shadowType => (app) => {
    const ground = new Entity('Ground');
    ground.addComponent('render', { type: 'plane', material: createMaterial(Color.WHITE) });
    ground.setLocalScale(16, 1, 16);
    app.root.addChild(ground);

    const box = new Entity('Box');
    box.addComponent('render', { type: 'box', material: createMaterial(Color.WHITE) });
    box.setPosition(0, 1, -1);
    box.setLocalScale(2, 2, 2);
    app.root.addChild(box);

    const sun = new Entity('Sun');
    sun.addComponent('light', {
        type: 'directional',
        castShadows: true,
        shadowType,
        shadowResolution: 1024,
        shadowDistance: 20,
        shadowBias: 0.2,
        normalOffsetBias: 0.05,
        numCascades: 1,
        intensity: 1
    });
    sun.setEulerAngles(-40, 0, 0);
    app.root.addChild(sun);

    return { position: [0, 7, 7], target: [0, 0, 0], camera: { nearClip: 0.1, farClip: 100 } };
};

// the number of pixels darker than a threshold, the shadowed pixels of the shadow scene
const countDark = (pixels) => {
    let count = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] + pixels[i + 1] + pixels[i + 2] < 120) {
            count++;
        }
    }
    return count;
};

// the mean absolute difference of two images, per channel
const meanDifference = (a, b) => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.abs(a[i] - b[i]);
    }
    return sum / a.length;
};

/**
 * Renders boxes at different distances, with the scene depth grabbed from the depth buffer of the
 * render target, and reads the depth of the whole view back.
 *
 * @param {boolean} reverseZ - Whether the device uses reverse-z.
 * @param {number} samples - The number of samples of the render target. With more than one, the
 * grab resolves the multisampled depth with the default DEPTHRESOLVE_MIN, the nearest surface.
 * @returns {Promise<Float32Array|null>} The depth of each pixel, in world units, or null when the
 * backend executes no GPU work.
 */
const readSceneDepth = async (reverseZ, samples) => {
    const canvas = document.createElement('canvas');
    const graphicsDevice = createGraphicsDevice(canvas, { reverseZ });
    const app = new Application(canvas, { graphicsDevice });

    const colorBuffer = new Texture(graphicsDevice, { name: 'Color', width: WIDTH, height: HEIGHT, format: PIXELFORMAT_RGBA8, mipmaps: false });
    // with a stencil, as the engine allocates the depth buffer of a render target with one on WebGPU,
    // and the depth grab copies to the format the target declares
    const renderTarget = new RenderTarget({ colorBuffer, depth: true, stencil: true, samples });

    const material = createMaterial(Color.WHITE, false);
    [[-1.5, 0, -4], [1.5, 0.5, -9], [0, -1.5, -20]].forEach(([x, y, z], i) => {
        const box = new Entity(`Box${i}`);
        box.addComponent('render', { type: 'box', material });
        box.setPosition(x, y, z);
        box.setEulerAngles(20 * i, 30, 10);
        app.root.addChild(box);
    });

    const camera = new Entity('Camera');
    camera.addComponent('camera', { renderTarget, nearClip: 0.5, farClip: 50 });
    app.root.addChild(camera);
    camera.camera.requestSceneDepthMap(true);

    const reader = new SceneDepthReader(camera.camera);

    // the reader reads once the camera has rendered a depth, and services the read as the camera
    // renders, landing a frame or two later
    app.render();
    let depths = null;
    reader.read(new Vec4(0, 0, 1, 1), WIDTH, HEIGHT).then((result) => {
        depths = result;
    });
    const renderFrames = async (count) => {
        if (depths || count === 0) {
            return;
        }
        app.render();
        await graphicsDevice.wgpu.queue.onSubmittedWorkDone();
        await new Promise((resolve) => {
            setImmediate(resolve);
        });
        await renderFrames(count - 1);
    };
    await renderFrames(20);

    reader.destroy();
    renderTarget.destroyTextureBuffers();
    renderTarget.destroy();
    app.destroy();
    return executesGpuWork ? depths : null;
};

const pixelAt = (pixels, x, y) => {
    const i = (y * WIDTH + x) * 4;
    return [pixels[i], pixels[i + 1], pixels[i + 2]];
};

describe('Reverse-z rendering', function () {

    beforeEach(function () {
        // reverse-z is a WebGPU feature, which the null device does not implement
        if (getTestDeviceType() !== 'webgpu') {
            this.skip();
        }
        jsdomSetup();
    });

    afterEach(function () {
        jsdomTeardown();
    });

    it('draws the nearest of overlapping surfaces, in front of the clear depth', async function () {
        const build = (app) => {
            const near = new Entity('Near');
            near.addComponent('render', { type: 'box', material: createMaterial(new Color(1, 0, 0), false) });
            near.setPosition(0, 0, -4);
            app.root.addChild(near);

            const far = new Entity('Far');
            far.addComponent('render', { type: 'box', material: createMaterial(new Color(0, 0, 1), false) });
            far.setPosition(0, 0, -8);
            far.setLocalScale(6, 6, 1);
            app.root.addChild(far);

            return { camera: { nearClip: 0.5, farClip: 50 } };
        };

        const standard = await renderScene(false, build);
        const reversed = await renderScene(true, build);
        if (executesGpuWork) {
            // the near box covers the center, the far one the area around it
            expect(pixelAt(reversed, 32, 32)).to.deep.equal([255, 0, 0]);
            expect(pixelAt(reversed, 32, 8)).to.deep.equal([0, 0, 255]);
            expect(meanDifference(standard, reversed)).to.be.below(0.5);
        }
    });

    it('resolves surfaces far from the camera, which the standard depth cannot', async function () {
        // two quads 1 unit apart at a distance of 100000, with a near clip of 0.1, rendered into a
        // 32-bit float depth buffer: the precision reverse-z exists for
        const build = (app) => {
            const mesh = Mesh.fromGeometry(app.graphicsDevice, new PlaneGeometry({ halfExtents: new Vec2(0.5, 0.5) }));
            const addQuad = (color, z, size) => {
                const material = createMaterial(color, false);
                material.cull = CULLFACE_NONE;
                material.update();
                const quad = new Entity();
                quad.addComponent('render', { meshInstances: [new MeshInstance(mesh, material)] });
                quad.setEulerAngles(90, 0, 0);
                quad.setLocalScale(size, size, size);
                quad.setPosition(0, 0, z);
                app.root.addChild(quad);
            };
            addQuad(new Color(0, 0, 1), -100001, 60000);
            addQuad(new Color(1, 0, 0), -100000, 30000);
            return { camera: { nearClip: 0.1, farClip: 1000000, fov: 60 } };
        };

        const depth32 = device => ({
            depthBuffer: new Texture(device, { name: 'Depth', width: WIDTH, height: HEIGHT, format: PIXELFORMAT_DEPTH, mipmaps: false })
        });
        const render = reverseZ => renderScene(reverseZ, build, depth32);

        const reversed = await render(true);
        if (executesGpuWork) {
            // the red quad in front covers the center, the blue quad behind it the area around it
            expect(pixelAt(reversed, 32, 32)).to.deep.equal([255, 0, 0]);
            expect(pixelAt(reversed, 32, 28)).to.deep.equal([255, 0, 0]);
            expect(pixelAt(reversed, 32, 20)).to.deep.equal([0, 0, 255]);
        }
    });

    for (const [name, shadowType] of [
        ['PCF1 32F', SHADOW_PCF1_32F],
        ['PCF3 16F', SHADOW_PCF3_16F],
        ['PCF5 32F', SHADOW_PCF5_32F],
        ['VSM 16F', SHADOW_VSM_16F],
        ['VSM 32F', SHADOW_VSM_32F],
        ['PCSS 32F', SHADOW_PCSS_32F]
    ]) {
        it(`renders the same directional ${name} shadow as the standard depth`, async function () {
            const standard = await renderScene(false, shadowScene(shadowType));
            const reversed = await renderScene(true, shadowScene(shadowType));
            if (executesGpuWork) {
                // the images match, and the scene renders a shadow, so the match covers the
                // shadow map rendering and its sampling
                expect(meanDifference(standard, reversed)).to.be.below(1);
                expect(countDark(standard)).to.be.above(300);
                expect(Math.abs(countDark(reversed) - countDark(standard))).to.be.below(20);
            }
        });
    }

    for (const samples of [1, 4]) {
        it(`reads back the same scene depth as the standard depth, with ${samples} sample(s)`, async function () {
            const standard = await readSceneDepth(false, samples);
            const reversed = await readSceneDepth(true, samples);
            if (executesGpuWork) {
                expect(reversed).to.be.an.instanceOf(Float32Array);
                let hits = 0;
                for (let i = 0; i < standard.length; i++) {
                    if (Number.isFinite(standard[i])) {
                        hits++;
                        // the reverse-z linearization keeps the precision, so the depths match closely
                        expect(reversed[i]).to.be.closeTo(standard[i], standard[i] * 1e-4);
                    } else {
                        // nothing rendered at the pixel, in both
                        expect(reversed[i]).to.equal(standard[i]);
                    }
                }
                expect(hits).to.be.above(200);
            }
        });
    }

    it('applies the same linear fog as the standard depth', async function () {
        const build = (app) => {
            app.scene.fog.type = FOG_LINEAR;
            app.scene.fog.color = new Color(0, 1, 0);
            app.scene.fog.start = 2;
            app.scene.fog.end = 30;

            const ground = new Entity('Ground');
            ground.addComponent('render', { type: 'plane', material: createMaterial(Color.WHITE, false) });
            ground.setLocalScale(100, 1, 100);
            app.root.addChild(ground);
            return { position: [0, 2, 0], target: [0, 0, -20], camera: { nearClip: 0.1, farClip: 100 } };
        };
        const standard = await renderScene(false, build);
        const reversed = await renderScene(true, build);
        if (executesGpuWork) {
            expect(meanDifference(standard, reversed)).to.be.below(1);
        }
    });
});
