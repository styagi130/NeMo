# EasyMagpie serving CPU regressions

Run in the pinned vLLM-Omni environment after installing this checkout's runtime
patches (see the package README):

```bash
python tools/easymagpie_vllm_omni/tests/run_cpu.py
```

The runner selects the CPU platform and hides CUDA before importing PyTorch.
It imports the plugin directly from this checkout; no model weights, benchmark
artifacts, network connection, or repository-wide NeMo installation is needed.
The installed vLLM/vLLM-Omni versions must match `runtime_patches/manifest.json`.

For the bounded scheduler/admission subset:

```bash
python tools/easymagpie_vllm_omni/tests/run_cpu.py tools/easymagpie_vllm_omni/tests/test_admission_fence.py tools/easymagpie_vllm_omni/tests/test_phase_admission.py tools/easymagpie_vllm_omni/tests/test_scheduler.py
```

The pool ownership and startup-policy tests also run with Python and pytest alone:

```bash
pytest --confcutdir=tools/easymagpie_vllm_omni/tests -o addopts= tools/easymagpie_vllm_omni/tests/test_dds_routing_pool.py tools/easymagpie_vllm_omni/tests/test_query_dispatch_pool.py tools/easymagpie_vllm_omni/tests/test_optimized_workers.py
```

Coverage includes the four-step global admission bound, cancellation/preemption
and reused request identities, async codec settlement and routing, immutable
feedback snapshots, pinned-pool retirement/quarantine, metadata and sender
fallback parity, and startup precision/prewarm ordering. Delayed CPU stream/event
stand-ins test pool ownership failures; they do not replace actual CUDA tests.
The copied optional CUDA feedback/decode cases are skipped by the CPU runner.
Numerical GPU parity, CUDA graph replay, first-use compilation, throughput and
ASR remain separate hardware validation gates; this suite makes no performance
or acoustic-equivalence claim. Artifact-manifest/report acceptance tests are not
part of the shipping behavior suite.
