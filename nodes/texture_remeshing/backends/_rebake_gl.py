# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Hardware (OpenGL) UV-space rasterization for the Rebake Texture GPU backend.

Same job as rebake_cpu.rasterize_uv_mesh (find every texel covered by uv_mesh's UV
triangles and its interpolated 3D surface position) but using the GPU's real hardware
rasterizer via a headless EGL context (moderngl), instead of a Python per-face loop.
Not the actual bottleneck in this node (closest-point search is), but real hardware
rasterization is the correct tool if it ever becomes one -- no differentiable-rendering
research library needed for a non-differentiable UV bake.

Trick: treat UV coordinates as clip-space xy (the standard "render in UV space" bake
technique) so the rasterizer answers "which triangle/position is under this texel" for
every texel at once, instead of one triangle at a time.
"""

import numpy as np

_VERTEX_SHADER = """
#version 330
// texture_size passed as a uniform so texel (px,py) lands EXACTLY where
// rebake_cpu.rasterize_uv_mesh would put it: fx=u*(T-1), fy=(1-v)*(T-1) --
// a "T points spanning [0,1] inclusive" grid, not GL's native half-open
// per-texel-center convention (px=u*T). Calibrated empirically against
// rebake_cpu's rasterizer (see nodes/texture_remeshing/backends/_rebake_gl.py
// dev notes) -- do not simplify this back to a plain uv*2-1 without re-verifying,
// a naive mapping is off by a sub-texel amount that silently corrupts every bake.
uniform float texture_size;
in vec2 in_uv;
in vec3 in_pos;
out vec3 v_pos;
void main() {
    float u = in_uv.x;
    float v = 1.0 - in_uv.y;
    float T = texture_size;
    float rx = 2.0 * (u * (T - 1.0) + 0.5) / T - 1.0;
    float ry = 2.0 * (v * (T - 1.0) + 0.5) / T - 1.0;
    gl_Position = vec4(rx, ry, 0.0, 1.0);
    v_pos = in_pos;
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 v_pos;
out vec4 f_color;
void main() {
    f_color = vec4(v_pos, 1.0);
}
"""

_ctx = None


def _get_context():
    """Lazily create (and cache) a single headless EGL context for this process."""
    global _ctx
    if _ctx is None:
        import moderngl
        _ctx = moderngl.create_context(standalone=True, backend='egl')
    return _ctx


def rasterize_uv_mesh_gl(uv_mesh, texture_size):
    """Rasterize every UV triangle of uv_mesh via the GPU hardware rasterizer.
    Returns parallel arrays: pixel x, pixel y (int, image coords -- same convention as
    rebake_cpu.rasterize_uv_mesh: v=0 is the texture's bottom row), and the interpolated
    3D surface position -- a drop-in replacement for the CPU rasterizer's output."""
    import moderngl

    if not hasattr(uv_mesh, 'visual') or not hasattr(uv_mesh.visual, 'uv') or uv_mesh.visual.uv is None:
        raise ValueError("uv_mesh has no UV coordinates -- run a UV Unwrap node on it first "
                          "(Xatlas, ARAP, LSCM, Harmonic, Geogram ABF, ...).")

    ctx = _get_context()
    T = int(texture_size)

    uvs = np.asarray(uv_mesh.visual.uv, dtype=np.float32)
    verts = np.asarray(uv_mesh.vertices, dtype=np.float32)
    faces = np.asarray(uv_mesh.faces, dtype=np.int32)

    prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
    prog['texture_size'].value = float(T)
    vbo_uv = ctx.buffer(np.ascontiguousarray(uvs).tobytes())
    vbo_pos = ctx.buffer(np.ascontiguousarray(verts).tobytes())
    ibo = ctx.buffer(np.ascontiguousarray(faces).tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo_uv, '2f', 'in_uv'), (vbo_pos, '3f', 'in_pos')],
        ibo,
    )

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((T, T), 4, dtype='f4')])
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 0.0)  # alpha=0 marks "uncovered"
    ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
    vao.render(moderngl.TRIANGLES)

    raw = fbo.color_attachments[0].read()
    img = np.frombuffer(raw, dtype=np.float32).reshape(T, T, 4)

    vao.release(); ibo.release(); vbo_pos.release(); vbo_uv.release(); prog.release(); fbo.release()

    covered = img[:, :, 3] > 0.5
    py, px = np.nonzero(covered)
    pos = img[py, px, :3].astype(np.float64)

    if len(px) == 0:
        raise ValueError("No texels rasterized -- UV layout may be degenerate or texture_size too small.")

    return px.astype(np.int64), py.astype(np.int64), pos
