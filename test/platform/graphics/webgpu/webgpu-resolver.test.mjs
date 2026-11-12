import { expect } from 'chai';

import { Preprocessor } from '../../../../src/core/preprocessor.js';
import { DEPTHRESOLVE_MAX, DEPTHRESOLVE_MIN, DEPTHRESOLVE_SAMPLE0 } from '../../../../src/platform/graphics/constants.js';
import webgpuDepthResolve from '../../../../src/platform/graphics/shader-chunks/frag/webgpu-depth-resolve.js';
import { WebgpuResolver } from '../../../../src/platform/graphics/webgpu/webgpu-resolver.js';

// mirrors how WebgpuResolver#getShader builds the shader source for a resolve mode, and how the
// Shader class preprocesses WGSL (stripDefines: true)
const process = (mode) => {
    const code = `#define DEPTH_RESOLVE_${mode.toUpperCase()}\n${webgpuDepthResolve}`;
    return Preprocessor.run(code, new Map(), { stripDefines: true });
};

describe('WebgpuResolver depth resolve shader chunk', function () {

    it('loads only sample 0 for the sample0 mode', function () {
        const code = process('sample0');
        expect(code).to.contain('textureLoad(img, coord, 0u)');
        expect(code).to.not.contain('textureNumSamples');
        expect(code).to.not.contain('#');
    });

    it('reduces all samples with min for the min mode', function () {
        const code = process('min');
        expect(code).to.contain('textureNumSamples(img)');
        expect(code).to.contain('min(depth, textureLoad(img, coord, i))');
        expect(code).to.not.contain('max(depth');
        expect(code).to.not.contain('#');
    });

    it('reduces all samples with max for the max mode', function () {
        const code = process('max');
        expect(code).to.contain('textureNumSamples(img)');
        expect(code).to.contain('max(depth, textureLoad(img, coord, i))');
        expect(code).to.not.contain('min(depth');
        expect(code).to.not.contain('#');
    });
});

describe('WebgpuResolver.getReduction', function () {

    it('reduces with the mode itself with a standard depth buffer', function () {
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_MIN, false)).to.equal(DEPTHRESOLVE_MIN);
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_MAX, false)).to.equal(DEPTHRESOLVE_MAX);
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_SAMPLE0, false)).to.equal(DEPTHRESOLVE_SAMPLE0);
    });

    it('swaps min and max with reverse-z, so that min still selects the nearest surface', function () {
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_MIN, true)).to.equal(DEPTHRESOLVE_MAX);
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_MAX, true)).to.equal(DEPTHRESOLVE_MIN);
        expect(WebgpuResolver.getReduction(DEPTHRESOLVE_SAMPLE0, true)).to.equal(DEPTHRESOLVE_SAMPLE0);
    });
});
