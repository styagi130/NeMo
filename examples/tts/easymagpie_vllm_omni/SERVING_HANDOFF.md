# EasyMagpie lightweight serving handoff

## Goal

Replace the Triton Python-backend/BLS serving path with a lightweight Python
service while retaining concurrent batching and streaming audio. The service
should improve GPU utilization, audio RTFX, and potentially TTFA by removing
the codec's CPU/GPU handoff.

## Recommended design

Run two transports in one process:

```text
HTTP clients  -> FastAPI  --+
                           +-> shared async TTS scheduler -> vLLM-Omni -> TensorRT codec
gRPC clients  -> grpc.aio --+
```

Use FastAPI/Uvicorn for REST, health, and metrics endpoints. Use `grpc.aio`
for low-latency streaming synthesis (and a Riva-compatible RPC surface if
needed). They must share exactly one model/engine set and one scheduler;
neither transport should own its own GPU model instance.

Suggested ports: HTTP `8000`, gRPC `50051`.

## Critical performance requirements

1. Keep codec IDs on the GPU after vLLM-Omni generation.
2. Invoke the TensorRT codec with GPU tensors, preferably through DLPack or
   the TensorRT Python API. Do not call `.cpu().numpy()` between the talker
   and codec.
3. Keep decoded audio on GPU until it is emitted to the response path; transfer
   only the final client payload when necessary.
4. Batch codec work in one central collector, keyed by generation step/window,
   across all HTTP and gRPC requests.
5. Use a low-latency policy for the first audio chunk and a throughput-oriented
   policy for later chunks. This makes TTFA a deliberate tradeoff rather than
   allowing arbitrary per-request fragmentation.

## Why this is needed

The traced Triton B32 configuration successfully formed frontend B32 requests
with a 20 ms queue delay. It still showed codec fragmentation and synchronization
overhead:

- Approximately 0.65 seconds of GPU kernel time over a 3.9 second trace span
  (~17% kernel-active time).
- 67 codec executions for 494 codec inputs, including 33 B1 executions.
- 33,032 `cudaMemcpyAsync` and 19,717 `cudaStreamSynchronize` calls.

The current backend converts generated IDs with `codes.detach().cpu().numpy()`,
submits codec BLS requests from Python, and converts codec output back to host
arrays. This is the primary target; increasing the frontend queue delay further
is unlikely to help now that frontend B32 formation is working.

Trace artifact:
`nsys_traces/triton_b32_20ms_serving.nsys-rep`.

## Implementation sketch

1. Create one `AsyncTTSScheduler` with a request queue and a GPU codec batch
   queue.
2. Start one embedded vLLM-Omni async engine and one TensorRT codec runner at
   process startup.
3. Implement FastAPI and `grpc.aio` handlers as thin adapters that enqueue a
   request and stream scheduler output.
4. In the scheduler, concatenate compatible GPU-resident codec windows into a
   codec batch, run one TensorRT enqueue, and scatter results back to streams.
5. Instrument queue wait, first-codec dispatch, TTFA, codec batch-size
   distribution, and RTFX before comparing it to Triton.

## Non-goals / cautions

- FastAPI or gRPC does not itself make inference faster; the shared scheduler
  and GPU-resident codec handoff do.
- Do not run separate FastAPI and gRPC model processes, which would duplicate
  GPU memory and split batches.
- TorchServe, Ray Serve, and KServe are not attractive replacements here: they
  add serving infrastructure without solving the codec boundary.
- BentoML can help package this later, but a direct Python service provides the
  best control for the initial performance implementation.
