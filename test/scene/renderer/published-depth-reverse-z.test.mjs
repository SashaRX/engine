import { expect } from 'chai';

import { Mat4 } from '../../../src/core/math/mat4.js';
import { Vec3 } from '../../../src/core/math/vec3.js';
import { Vec4 } from '../../../src/core/math/vec4.js';
import { Entity } from '../../../src/framework/entity.js';
import { ComputePass } from '../../../src/platform/graphics/compute-pass.js';
import { Compute } from '../../../src/platform/graphics/compute.js';
import {
    BUFFERUSAGE_COPY_SRC, FUNC_ALWAYS, PIXELFORMAT_R32F, SEMANTIC_POSITION, SHADERLANGUAGE_WGSL
} from '../../../src/platform/graphics/constants.js';
import { DepthState } from '../../../src/platform/graphics/depth-state.js';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { ScopeSpace } from '../../../src/platform/graphics/scope-space.js';
import { Shader } from '../../../src/platform/graphics/shader.js';
import { StorageBuffer } from '../../../src/platform/graphics/storage-buffer.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { Camera } from '../../../src/scene/camera.js';
import { ASPECT_MANUAL, PROJECTION_ORTHOGRAPHIC, PROJECTION_PERSPECTIVE } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import { RenderPassShaderQuad } from '../../../src/scene/graphics/render-pass-shader-quad.js';
import { ShaderUtils } from '../../../src/scene/shader-lib/shader-utils.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// Independent reference: remap the rows of the standard projection, without calling
// applyShaderProjectionTransform or _frameViewProjection (the code under test).
const referenceViewProjection = (camera, flipY) => {
    const projection = camera.projectionMatrix.clone();
    const data = projection.data;
    for (let column = 0; column < 4; column++) {
        const i = column * 4;
        if (flipY) data[i + 1] *= -1;
        if (camera.device.isWebGPU) {
            data[i + 2] = 0.5 * (data[i + 3] + (camera.reverseZActive ? -data[i + 2] : data[i + 2]));
        }
    }
    const world = new Mat4().setTRS(camera.node.getPosition(), camera.node.getRotation(), Vec3.ONE);
    return projection.mul(new Mat4().invert(world));
};

const unproject = (inverse, x, y, depth) => {
    const p = inverse.transformVec4(new Vec4(x, y, depth, 1));
    return new Vec3(p.x / p.w, p.y / p.w, p.z / p.w);
};

const expectMatrix = (actual, expected) => {
    actual.data.forEach((value, i) => {
        expect(value, `matrix[${i}]`).to.be.closeTo(expected.data[i], 1e-6 * Math.max(1, Math.abs(expected.data[i])));
    });
};

describe('Published depth / reverse-z compatibility', function () {
    describe('CPU matrix and depth round trips (no GPU)', function () {
        for (const webgpu of [false, true]) {
            for (const projection of [PROJECTION_PERSPECTIVE, PROJECTION_ORTHOGRAPHIC]) {
                for (const reverseZ of (webgpu ? [false, true] : [false])) {
                    for (const flipY of [false, true]) {
                        it(`webgpu=${webgpu}, projection=${projection}, reverseZ=${reverseZ}, flipY=${flipY}`, function () {
                            // A capability-only fixture: this is explicitly CPU mathematics, not a WebGPU device.
                            const camera = new Camera({ isWebGPU: webgpu, scope: new ScopeSpace('matrix-test') });
                            camera.node = new GraphNode();
                            camera.aspectRatioMode = ASPECT_MANUAL;
                            camera.aspectRatio = 1;
                            camera.nearClip = 0.5;
                            camera.farClip = 1000;
                            camera.projection = projection;
                            camera.reverseZ = reverseZ;
                            const texture = { width: 32, height: 32 };
                            let previous;
                            for (let frame = 1; frame <= 2; frame++) {
                                camera.node.setPosition(frame === 1 ? 0 : 10, frame === 1 ? 0 : -3, frame === 1 ? 0 : 7);
                                camera.node.setEulerAngles(frame === 1 ? 0 : 11, frame === 1 ? 0 : 23, 0);
                                expect(camera._shaderMatricesVersion).to.not.equal(frame);
                                camera.publishSceneDepthMap(texture, frame, false, false, false, flipY);
                                const published = camera.sceneDepthMapHandle.viewProjectionInverse.clone();
                                const vp = referenceViewProjection(camera, flipY);
                                expectMatrix(published, new Mat4().invert(vp));
                                if (previous) expect(published.equals(previous)).to.equal(false);

                                const world = new Mat4().setTRS(camera.node.getPosition(), camera.node.getRotation(), Vec3.ONE);
                                for (const distance of [0.5, 3, 50, 900, 1000]) {
                                    const point = world.transformPoint(new Vec3(0.1, -0.1, -distance));
                                    const clip = vp.transformVec4(new Vec4(point.x, point.y, point.z, 1));
                                    const storedDepth = Math.fround(webgpu ? clip.z / clip.w : 0.5 * (clip.z / clip.w + 1));
                                    const reconstructed = unproject(published, clip.x / clip.w, clip.y / clip.w,
                                        webgpu ? storedDepth : 2 * storedDepth - 1);
                                    // Float32 forward depth loses precision at the far plane; retain a distance-scaled bound.
                                    expect(reconstructed.distance(point), `frame=${frame}, distance=${distance}`)
                                    .to.be.lessThan(2e-4 * Math.max(1, distance));
                                }
                                // Simulate the subsequent main render so the second frame must reject its stale matrices.
                                camera._storeShaderMatrices(vp, 0, 0, frame);
                                camera.publishSceneDepthMap(texture, frame, false, false, false, flipY);
                                expectMatrix(camera.sceneDepthMapHandle.viewProjectionInverse, published);
                                previous = published;
                            }
                            camera.destroy();
                        });
                    }
                }
            }
        }

        it('uses reverseZActive, including the XR exclusion', function () {
            const camera = new Camera({ isWebGPU: true, scope: new ScopeSpace('xr-matrix-test') });
            camera.node = new GraphNode();
            camera.reverseZ = true;
            camera.xrViews = [];
            expect(camera.reverseZActive).to.equal(false);
            camera.publishSceneDepthMap({ width: 4, height: 4 }, 1, false, false, false);
            expectMatrix(camera.sceneDepthMapHandle.viewProjectionInverse, new Mat4().invert(referenceViewProjection(camera, false)));
            camera.destroy();
        });
    });

    describe('WebGPU application pre-pass and compute consumer', function () {
        let app;
        let resources;

        beforeEach(function () {
            jsdomSetup();
            app = createApp();
            resources = [];
            if (!app.graphicsDevice.isWebGPU) this.skip();
        });

        afterEach(function () {
            for (const resource of resources.reverse()) resource.destroy();
            app?.destroy();
            jsdomTeardown();
        });

        [false, true].forEach((reverseZ) => {
            it(`publishes and reconstructs rendered raw depth on two moving frames, reverseZ=${reverseZ}`, async function () {
                const device = app.graphicsDevice;
                const own = (resource) => {
                    resources.push(resource);
                    return resource;
                };
                const entity = new Entity('depth-camera');
                entity.addComponent('camera', {
                    reverseZ,
                    nearClip: 0.5,
                    farClip: 10000,
                    fov: 60,
                    aspectRatioMode: ASPECT_MANUAL,
                    aspectRatio: 1
                });
                app.root.addChild(entity);
                const component = entity.camera;
                const camera = component.camera;
                const depth = own(new Texture(device, { width: 3, height: 1, format: PIXELFORMAT_R32F, mipmaps: false }));
                const target = own(new RenderTarget({ colorBuffer: depth, depth: true, samples: 1 }));
                const producerShader = own(ShaderUtils.createShader(device, {
                    uniqueName: 'PublishedDepthCompatibilityProducer',
                    attributes: { aPosition: SEMANTIC_POSITION },
                    vertexChunk: 'quadVS',
                    fragmentWGSL: `
                        uniform producerProjection: mat4x4f;
                        @fragment fn fragmentMain(input: FragmentInput) -> FragmentOutput {
                            var output: FragmentOutput;
                            let distances = array<f32, 3>(2.0, 50.0, 9000.0);
                            let distance = distances[u32(input.position.x)];
                            let clip = uniform.producerProjection * vec4f(0.0, 0.0, -distance, 1.0);
                            let depth = clip.z / clip.w;
                            output.fragDepth = depth;
                            output.color = vec4f(depth, 0.0, 0.0, 1.0);
                            return output;
                        }
                    `
                }));
                // This application pre-pass writes raw depth both to a depth attachment and R32F.
                // It deliberately does not call setCameraUniforms, which would populate the stored-matrix fast path.
                const prepass = own(new RenderPassShaderQuad(device));
                prepass.shader = producerShader;
                prepass.depthState = new DepthState(FUNC_ALWAYS, true);
                prepass.init(target);
                prepass.setClearDepth(reverseZ ? 0 : 1);
                const computeShader = own(new Shader(device, {
                    name: 'PublishedDepthCompatibilityConsumer',
                    shaderLanguage: SHADERLANGUAGE_WGSL,
                    cshader: `
                        #include "sceneDepthCS"
                        var<storage, read_write> results: array<vec4f>;
                        @compute @workgroup_size(1)
                        fn main(@builtin(global_invocation_id) id: vec3u) {
                            let texel = vec2i(i32(id.x), 0);
                            results[id.x] = vec4f(sceneDepthWorldPosition(texel), textureLoad(computeSceneDepthMap, texel, 0).r);
                        }
                    `
                }));
                const results = own(new StorageBuffer(device, 3 * 16, BUFFERUSAGE_COPY_SRC));
                const compute = own(new Compute(device, computeShader, 'DepthCompatibility'));
                compute.setSceneDepthMap(component.sceneDepthMapHandle);
                compute.setParameter('results', results);
                compute.setupDispatch(3, 1, 1);
                const consumer = own(new ComputePass(device, [compute]));
                let published;
                let reference;
                let frameVersion;
                prepass.onBefore = () => {
                    frameVersion = device.renderVersion;
                    expect(camera._shaderMatricesVersion, 'must exercise the pre-render fallback').to.not.equal(frameVersion);
                    reference = referenceViewProjection(camera, target.flipY);
                    const projection = Camera.applyShaderProjectionTransform(camera.projectionMatrix, new Mat4(),
                        target.flipY, true, camera.reverseZActive);
                    device.scope.resolve('producerProjection').setValue(projection.data);
                };
                prepass.onAfter = () => {
                    component.publishSceneDepthMap(depth, { flipY: target.flipY });
                    published = component.sceneDepthMapHandle.viewProjectionInverse.clone();
                };
                consumer.onBefore = () => {
                    expect(component.sceneDepthMapHandle.texture).to.equal(depth);
                    expect(component.sceneDepthMapHandle.linear).to.equal(false);
                    expect(component.sceneDepthMapHandle.renderVersion).to.equal(frameVersion);
                    expect(camera._shaderMatricesVersion, 'compute precedes the main camera render').to.not.equal(frameVersion);
                };
                component.beforePasses.push(prepass, consumer);

                let first;
                const renderFrame = async (frame) => {
                    entity.setPosition(frame * 10, frame * -3, frame * 7);
                    entity.setEulerAngles(frame * 11, frame * 23, 0);
                    app.render();
                    if (first) expect(published.equals(first)).to.equal(false);
                    first = published.clone();

                    // Dawn null validates API/shaders but does not execute them. Never count its zero readbacks as results.
                    if (process.env.PC_WEBGPU_BACKEND && process.env.PC_WEBGPU_BACKEND !== 'null') {
                        const values = new Float32Array(12);
                        await results.read(0, values.byteLength, values, true);
                        const depths = await depth.read(0, 0, 3, 1, { immediate: true });
                        const distances = [2, 50, 9000];
                        const world = new Mat4().setTRS(entity.getPosition(), entity.getRotation(), Vec3.ONE);
                        const observations = [];
                        for (let x = 0; x < 3; x++) {
                            const distance = distances[x];
                            const ndcX = 2 * (x + 0.5) / 3 - 1;
                            const point = world.transformPoint(new Vec3(ndcX * distance / camera.projectionMatrix.data[0], 0, -distance));
                            const actual = new Vec3(values[4 * x], values[4 * x + 1], values[4 * x + 2]);
                            const error = actual.distance(point);
                            expect(depths[x], `depth texel ${x}`).to.be.greaterThan(0).and.lessThan(1);
                            expect(values[4 * x + 3]).to.equal(depths[x]);
                            observations.push({ reverseZ, frame, distance, storedDepth: depths[x], worldError: error });
                        }
                        console.log(JSON.stringify(observations));
                        for (const sample of observations) {
                            const tolerance = (reverseZ ? 1e-6 : 0.002) * sample.distance;
                            expect(sample.worldError, `frame=${frame}, distance=${sample.distance}`).to.be.lessThan(tolerance);
                        }
                        expect(depths[0] > depths[2]).to.equal(reverseZ);
                    }
                    expectMatrix(published, new Mat4().invert(reference));
                    expectMatrix(published, camera._viewProjInverse);
                };
                await renderFrame(0);
                await renderFrame(1);
            });
        });
    });
});
