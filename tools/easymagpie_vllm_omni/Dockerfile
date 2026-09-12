# Build from the Speech repository root.
FROM vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52 AS cairo_builder
RUN apt-get update && apt-get install -y --no-install-recommends --no-remove \
    libcairo2-dev=1.16.0-5ubuntu2.1 pkg-config=0.29.2-1ubuntu3
RUN python3 -m pip wheel --no-cache-dir --no-index --no-deps --no-build-isolation \
    --wheel-dir /wheels \
    'https://files.pythonhosted.org/packages/19/4f/0d48a017090d4527e921d6892bc550ae869902e67859fc960f8fe63a9094/pycairo-1.26.1.tar.gz#sha256=a11b999ce55b798dbf13516ab038e0ce8b6ec299b208d7c4e767a6f7e68e8430'

FROM vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52

COPY tools/easymagpie_vllm_omni/requirements.txt /tmp/easymagpie-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/easymagpie-requirements.txt

COPY --from=cairo_builder /wheels /wheels
RUN python3 -m pip install --no-cache-dir --no-deps /wheels/pycairo-*.whl nixl-cu13==1.3.1

COPY tools/easymagpie_vllm_omni /opt/easymagpie
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/easymagpie \
    && python3 -m pip check

ENV XDG_CACHE_HOME=/tmp/easymagpie-cache \
    VLLM_CACHE_ROOT=/tmp/easymagpie-vllm \
    TORCHINDUCTOR_CACHE_DIR=/tmp/easymagpie-inductor \
    TRITON_CACHE_DIR=/tmp/easymagpie-triton \
    FLASHINFER_WORKSPACE_BASE=/tmp/easymagpie-flashinfer \
    HF_HOME=/tmp/easymagpie-hf \
    SPEAKER_SAMPLES_DIR=/tmp/easymagpie-speaker-samples

WORKDIR /opt/easymagpie
ENTRYPOINT ["bash", "/opt/easymagpie/scripts/run_server.sh"]
