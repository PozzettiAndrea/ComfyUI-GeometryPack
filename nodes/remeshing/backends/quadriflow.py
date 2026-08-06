# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""QuadriFlow quad remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshQuadriFlowNode(io.ComfyNode):
    """QuadriFlow quad remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_QuadriFlow",
            display_name="Remesh QuadriFlow (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=5000, min=100, max=5000000, step=100, tooltip="Target number of output faces (quad-dominant). QuadriFlow hits this fairly accurately (unlike Instant Meshes)."),
                io.Combo.Input("preserve_sharp", options=["true", "false"], default="false", tooltip="Align quads to sharp edges and keep them crisp. QuadriFlow's sharp threshold is HARDCODED at 60 deg (an edge is 'sharp' when adjacent face normals deviate > 60 deg) -- not adjustable from the binding. Turn ON for CAD/mechanical parts."),
                io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Keep the mesh boundary/open edges fixed (for open meshes)."),
                io.Combo.Input("adaptive_scale", options=["false", "true"], default="false", tooltip="Curvature-adaptive quad sizing: smaller quads where curvature is high (edges/fillets), larger on flats -- instead of a uniform grid. Great for CAD; spends faces where detail is."),
                io.Combo.Input("minimum_cost_flow", options=["false", "true"], default="false", tooltip="Use the min-cost-flow solver for the integer step -> cleaner quad connectivity and better-placed singularities. Slower but noticeably more regular output."),
                io.Combo.Input("aggressive_sat", options=["false", "true"], default="false", tooltip="Use the SAT solver for a fully-integer, seamless result with the fewest singularities (highest quality). Slowest; can be heavy on large meshes."),
                io.Int.Input("seed", default=0, min=0, max=2000000000, step=1, tooltip="Random seed for the field initialization (reproducible results)."),
                # QuadriFlow's cost scales with the INPUT mesh (orientation +
                # position fields are solved over every input face), not the
                # target. Decimating a dense input to ~2-3x the target first is
                # the standard speedup and barely affects output, since
                # QuadriFlow discards input connectivity anyway. Measured:
                # 112k-face input -> 20k target ran 6+ min single-core with all
                # fast settings; the field solve is the whole bill.
                #
                # PLAIN inputs here, deliberately -- not a DynamicCombo. This
                # backend node is reached via the dispatcher's GraphBuilder
                # expansion, and a dynamic input cannot be fed a literal value
                # across that boundary: the expander compares the live value
                # against the option keys ("on"/"off"), an assembled dict
                # matches neither, and the input is silently dropped -- the
                # user's toggle arrived here as None/off. The conditional
                # show-only-when-on UI lives on the dispatcher (remesh.py),
                # which unpacks its dict into these two scalars.
                io.Boolean.Input("pre_decimate", default=False, optional=True, tooltip="Quadric-collapse the input before remeshing. Big speedup on dense inputs; output quality is nearly unchanged because QuadriFlow rebuilds topology from scratch."),
                io.Int.Input("pre_decimate_faces", default=40000, min=1000, max=5000000, step=1000, optional=True, tooltip="Decimate the input to this many faces before QuadriFlow runs. Rule of thumb: 2-3x target_face_count. No-op if the input is already at or below this count."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=5000, preserve_sharp="false", preserve_boundary="true",
                adaptive_scale="false", minimum_cost_flow="false", aggressive_sat="false", seed=0,
                pre_decimate=None, pre_decimate_faces=None):
        from pyquadriflow.quadriflow import quadriflow_remesh

        # pre_decimate arrives as a plain bool (this node's own schema), but
        # tolerate every shape it has historically been sent in: the
        # dispatcher's DynamicCombo dict, a bare "on"/"off" string, or None
        # from workflows saved before the input existed.
        if isinstance(pre_decimate, dict):
            sel = pre_decimate.get("pre_decimate")
            pre_on = sel is True or sel == "on" or sel == "true"
            pre_faces = int(pre_decimate.get("pre_decimate_faces",
                                             pre_decimate_faces or 40000))
        else:
            pre_on = (pre_decimate is True or pre_decimate == "on"
                      or pre_decimate == "true")
            pre_faces = int(pre_decimate_faces or 40000)

        log.info("Backend: quadriflow")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        # pre_decimate is ALWAYS printed, off included: its absence from this
        # line is indistinguishable from the feature not being wired in (a
        # workflow saved before the toggle existed sends nothing and runs
        # with it off -- that should be visible, not silent).
        log.info("Parameters: target_face_count=%s, preserve_sharp=%s, preserve_boundary=%s, "
                 "adaptive_scale=%s, minimum_cost_flow=%s, aggressive_sat=%s, seed=%d, "
                 "pre_decimate=%s",
                 f"{target_face_count:,}", preserve_sharp, preserve_boundary,
                 adaptive_scale, minimum_cost_flow, aggressive_sat, seed,
                 f"on({pre_faces:,})" if pre_on else "off")
        pre_applied = False
        if pre_on and len(trimesh.faces) > pre_faces:
            import fast_simplification
            reduction = 1.0 - (pre_faces / len(trimesh.faces))
            log.info("Pre-decimate: %d -> ~%d faces (reduction=%.3f)...",
                     len(trimesh.faces), pre_faces, reduction)
            v_out, f_out = fast_simplification.simplify(
                np.asarray(trimesh.vertices, dtype=np.float32),
                np.asarray(trimesh.faces, dtype=np.int32),
                target_reduction=reduction,
            )
            src_mesh = trimesh_module.Trimesh(
                vertices=v_out, faces=f_out, process=False)
            log.info("Pre-decimate done: %d vertices, %d faces",
                     len(src_mesh.vertices), len(src_mesh.faces))
            pre_applied = True
        else:
            if pre_on:
                log.info("Pre-decimate: input (%d faces) already <= %d, skipping",
                         len(trimesh.faces), pre_faces)
            src_mesh = trimesh

        V = np.asarray(src_mesh.vertices, dtype=np.float64)
        F = np.asarray(src_mesh.faces, dtype=np.int32)

        out_vertices, out_faces = quadriflow_remesh(
            V, F, target_face_count,
            seed=int(seed),
            preserve_sharp=preserve_sharp == "true",
            preserve_boundary=preserve_boundary == "true",
            adaptive_scale=adaptive_scale == "true",
            minimum_cost_flow=minimum_cost_flow == "true",
            aggressive_sat=aggressive_sat == "true",
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.asarray(out_vertices, dtype=np.float32),
            faces=np.asarray(out_faces, dtype=np.int32),
            process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadriflow',
            'target_face_count': target_face_count,
            'preserve_sharp': preserve_sharp == "true",
            'preserve_boundary': preserve_boundary == "true",
            'adaptive_scale': adaptive_scale == "true",
            'minimum_cost_flow': minimum_cost_flow == "true",
            'aggressive_sat': aggressive_sat == "true",
            'seed': int(seed),
            'pre_decimate': pre_applied,
            'pre_decimate_faces': pre_faces if pre_applied else None,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (QuadriFlow): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"target_faces={target_face_count:,}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_QuadriFlow": RemeshQuadriFlowNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_QuadriFlow": "Remesh QuadriFlow (backend)"}
