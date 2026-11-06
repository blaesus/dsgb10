# Changes

## GLP steering (branch glp/steer-v1)

| # | Change | Default | Lossless | Before -> after |
|---|---|---|---|---|
| 11 | GLP spec-v1 steering at `residual_stream_post_layer` (`WEIGHTLESS_STEER_PATH`) | off | off: no launch, no read; `alpha 0` is the identity | not measured yet |

A GLP file holds one unit direction per decoder layer. At the end of layer L, after its write has been
folded into the four hc streams (`h = bf16(post[L] sub + sum_m comb[m] res[m])`), each stream is projected
off `direction.L`:

    h[n] <- bf16(h[n] - alpha * (h[n] . d) * d),   n = 0..3

which is the operation Owen's engine applies in `models/deepseek_v4_1/glp.py` + `tilelang/glp.py`
(spec v1, hook `residual_stream_post_layer`, `alpha` from `glp.alpha_default`, layers from the file). The
direction is applied after layer L and left of layer 0 unsteered, i.e. `direction.1..39` steer layers 1..39.

Where it runs: the decode/verify step steers the streams `ops::allreduce_hc` materialised for the next block
and re-collapses `yraw` from the projected streams with the same `pre` (so the next sublayer's norm, and the
head's final collapse on the last layer, read the steered stream; DSpark's `spec_hidden` and an Engram
layer's write already come after it); a prefill chunk splits the boundary in two, `hc_post` -> `glp_steer` ->
the next sublayer's `hc_pre` + norm + mixes, so both paths steer the same point of the forward.

Off unless `WEIGHTLESS_STEER_PATH` names a GGUF v3 file, which is read once at startup with the same
fail-closed checks Owen's reader makes: `glp.mode == "project"`, `glp.spec_version == 1`, rank 1, hook
`residual_stream_post_layer`, no `glp.dir_scales`/`glp.layer_scales`, one `direction.<L>` per layer of width
5120 (1-D F32, finite, unit norm within 1e-3), a layer list and `glp.content_sha256` that agree with the
tensors, and nothing but alignment padding after the last tensor. `WEIGHTLESS_STEER_ALPHA` and
`WEIGHTLESS_STEER_LAYERS` override the file and are refused when malformed. A file the reader cannot apply
exactly stops the rank at startup instead of serving unsteered.

`engine/model/tests/glp_test.cpp` (host only, no GPU) checks the reader against the real GLP-39 export
(39 directions, layers 1..39, alpha 0.75, worst |norm-1| 2.9e-8), six synthetic refusals, the env contract
(unset = off, overrides applied and refused), `alpha 0` bit identity and the projection's orthogonality.
