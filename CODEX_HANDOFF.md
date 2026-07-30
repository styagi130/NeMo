# Codex handoff

The H100 EasyMagpie optimization pass is complete. The measured outcome,
controlled comparisons, cold/outlier separation, trace findings, quality
results, and artifact paths are recorded in
[`H100_OPTIMIZATION_RESULTS.md`](H100_OPTIMIZATION_RESULTS.md).

Use
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_bs16.lock.yaml`
as the H100 reproducibility source of truth. It selects shared-expert stream 0,
the exact `NVIDIA_H100_NVL` MoE tile JSON, and 75% Stage-1 MPS.

The source checkout arrived with broad transferred mode-only changes and many
untracked EasyMagpie artifacts. Those unrelated changes were preserved. The
pre-existing A4500 lock and tile were also preserved byte-for-byte.

The host did not provide Lustre, so the run used the persistent local ext4
cache at `/workspace/.cache/easymp_h100`. `TMPDIR` was never set or changed.
