// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Rust -> vLLM-Omni EngineCore transport proof of concept.
//!
//! This intentionally has no HTTP server, Python bridge, or Riva text
//! normalization. It can load the checkpoint's Hugging Face tokenizer directly
//! in Rust, owns the vLLM startup handshakes, submits requests over ZMQ, and
//! consumes token, acoustic-code, and decoded-audio deltas.

use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use clap::Parser;
use rayon::prelude::*;
use rmpv::Value;
use serde::de::IgnoredAny;
use serde::{Deserialize, Deserializer};
use serde_tuple::Deserialize_tuple;
use tokio::time::timeout;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSendHalf, RouterSocket, ZmqMessage};

const REQUEST_TYPE_ADD: &[u8] = b"\x00";
const REQUEST_TYPE_ABORT: &[u8] = b"\x01";
const ENGINE_CORE_DEAD: &[u8] = b"ENGINE_CORE_DEAD";

#[derive(Debug, Parser)]
#[command(about = "Call EasyMagpie vLLM-Omni EngineCore processes directly from Rust")]
struct Args {
    /// Address on which Rust waits for StageEngineCoreProc HELLO/READY.
    #[arg(long, default_value = "tcp://127.0.0.1:62100")]
    handshake_address: String,

    /// Stage-1 address used with --full-pipeline.
    #[arg(long, default_value = "tcp://127.0.0.1:62101")]
    codec_handshake_address: String,

    /// Drive both the talker and stateful codec stages and emit a WAV.
    #[arg(long)]
    full_pipeline: bool,

    /// PCM16 WAV written by --full-pipeline.
    #[arg(long, default_value = "/tmp/easymagpie-rust.wav")]
    wav_output: PathBuf,

    /// Fallback codec output sample rate when no `sr` tensor is returned.
    #[arg(long, default_value_t = 22_050)]
    sample_rate: u32,

    /// Host advertised to the colocated Python engine for data-plane sockets.
    #[arg(long, default_value = "127.0.0.1")]
    advertised_host: String,

    /// Raw text passed in EasyMagpie additional_information (no normalization).
    #[arg(long, default_value = "Hello from the Rust vLLM engine client.")]
    text: String,

    /// Optional '<utterance-id>\t<text>' corpus. Each measured round uses
    /// Python-compatible random.choices selection with seed + round index.
    #[arg(long)]
    text_file: Option<PathBuf>,

    /// First measured-round corpus selection seed.
    #[arg(long, default_value_t = 20_260_729)]
    seed: u64,

    #[arg(long, default_value = "[EN]")]
    context_text: String,

    /// Enable direct Rust tokenization from this converted EasyMagpie model
    /// directory. The directory must contain tokenizer.json and config.json.
    #[arg(long)]
    rust_tokenizer_model: Option<PathBuf>,

    /// Print Rust-produced context and target token IDs, then exit without
    /// connecting to an engine. Requires --rust-tokenizer-model.
    #[arg(long, requires = "rust_tokenizer_model")]
    tokenize_only: bool,

    /// Corpus passes performed by --tokenize-only. One pass encodes every
    /// line in --text-file, or --text once when no corpus is supplied.
    #[arg(long, default_value_t = 1, requires = "tokenize_only")]
    tokenize_iterations: usize,

    /// Untimed corpus passes before a --tokenize-only measurement.
    #[arg(long, default_value_t = 0, requires = "tokenize_only")]
    tokenize_warmup_iterations: usize,

    /// Use the same encode_batch path used by concurrent inference cohorts.
    #[arg(long, requires = "tokenize_only")]
    tokenize_batch: bool,

    /// Expand/cycle the tokenizer-only corpus to this exact cohort size.
    #[arg(long, requires = "tokenize_only")]
    tokenize_batch_size: Option<usize>,

    #[arg(long, default_value = "eng")]
    speaker_id: String,

    /// Prompt length for the selected known-speaker checkpoint prompt.
    #[arg(long, default_value_t = 67)]
    prompt_len: usize,

    /// Backbone decode steps. Keep this small for a transport smoke test.
    #[arg(long, default_value_t = 32)]
    max_tokens: u32,

    /// Requests submitted together in each wave.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Untimed waves run before measurement.
    #[arg(long, default_value_t = 0)]
    warmup_rounds: usize,

    /// Timed waves run after warmup.
    #[arg(long, default_value_t = 1)]
    rounds: usize,

    /// Model frames before speech begins; used only for RTFX reporting.
    #[arg(long, default_value_t = 5)]
    speech_delay: usize,

    /// Codec frames represented by one generated model frame.
    #[arg(long, default_value_t = 2)]
    frame_stacking_factor: usize,

    /// Codec frame rate used to estimate audio duration.
    #[arg(long, default_value_t = 25.0)]
    codec_frame_rate: f64,

    /// Optional backbone stop token. Omit to force a length-limited smoke test.
    #[arg(long)]
    stop_token_id: Option<u32>,

    /// EasyMagpie local-transformer temperature.
    #[arg(long, default_value_t = 0.7)]
    audio_temperature: f64,

    /// EasyMagpie local-transformer top-k.
    #[arg(long, default_value_t = 80)]
    audio_top_k: u64,

    #[arg(long, default_value_t = 300)]
    ready_timeout_secs: u64,

    #[arg(long, default_value_t = 120)]
    output_timeout_secs: u64,

    /// Save raw multimodal tensor deltas and small metadata files here.
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct EasyMagpieModelConfig {
    text_eos_id: Option<u32>,
    text_vocab_size: Option<u32>,
    vocab_size: Option<u32>,
}

struct RustTextTokenizer {
    tokenizer: tokenizers::Tokenizer,
    context_ids: Vec<u32>,
    text_eos_id: u32,
}

struct RequestTokenIds<'a> {
    context_ids: &'a [u32],
    text_ids: Vec<u32>,
}

#[derive(Clone, Copy)]
struct RequestTokenIdSlices<'a> {
    context_ids: &'a [u32],
    text_ids: &'a [u32],
}

impl<'a> RequestTokenIds<'a> {
    fn as_slices(&'a self) -> RequestTokenIdSlices<'a> {
        RequestTokenIdSlices {
            context_ids: self.context_ids,
            text_ids: &self.text_ids,
        }
    }
}

impl RustTextTokenizer {
    fn load(model_dir: &Path, context_text: &str) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            anyhow!("load Rust tokenizer {}: {error}", tokenizer_path.display())
        })?;
        let context_ids = tokenizer
            .encode(context_text, true)
            .map_err(|error| anyhow!("tokenize context text in Rust: {error}"))?
            .get_ids()
            .to_vec();

        let config_path = model_dir.join("config.json");
        let config: EasyMagpieModelConfig = serde_json::from_slice(
            &fs::read(&config_path)
                .with_context(|| format!("read EasyMagpie config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse EasyMagpie config {}", config_path.display()))?;
        let text_eos_id = config
            .text_eos_id
            .or_else(|| {
                config
                    .text_vocab_size
                    .and_then(|vocabulary_size| vocabulary_size.checked_sub(2))
            })
            .or_else(|| {
                config
                    .vocab_size
                    .and_then(|vocabulary_size| vocabulary_size.checked_sub(2))
            })
            .context("config.json has no usable text_eos_id or text vocabulary size")?;

        Ok(Self {
            tokenizer,
            context_ids,
            text_eos_id,
        })
    }

    fn encode_request<'a>(&'a self, text: &str) -> Result<RequestTokenIds<'a>> {
        let mut text_ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow!("tokenize target text in Rust: {error}"))?
            .get_ids()
            .to_vec();
        text_ids.push(self.text_eos_id);
        Ok(RequestTokenIds {
            context_ids: &self.context_ids,
            text_ids,
        })
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(|error| anyhow!("tokenize target-text batch in Rust: {error}"))?;
        Ok(encodings
            .into_iter()
            .map(|encoding| {
                let mut ids = encoding.get_ids().to_vec();
                ids.push(self.text_eos_id);
                ids
            })
            .collect())
    }
}

struct ConnectedEngine {
    identity: Bytes,
    input_send: RouterSendHalf,
    output_socket: PullSocket,
}

#[derive(Debug)]
struct TensorDelta<'a> {
    key: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: &'a [u8],
}

#[derive(Debug)]
struct OwnedTensor {
    key: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct OutputEvent {
    request_id: String,
    new_tokens: usize,
    tensor_deltas: usize,
    finished: bool,
    audio_sample_counts: Vec<usize>,
    sample_rate: Option<u32>,
    tensors: Vec<OwnedTensor>,
}

enum WireTensorData {
    AuxIndex(usize),
    Raw(Vec<u8>),
}

impl<'de> Deserialize<'de> for WireTensorData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::Integer(index) => index
                .as_u64()
                .map(|index| Self::AuxIndex(index as usize))
                .ok_or_else(|| serde::de::Error::custom("tensor frame index must be non-negative")),
            Value::Ext(3, bytes) => Ok(Self::Raw(bytes)),
            Value::Ext(tag, _) => Err(serde::de::Error::custom(format!(
                "unsupported tensor extension type {tag}"
            ))),
            value => Err(serde::de::Error::custom(format!(
                "expected tensor frame index or raw-view extension, got {value:?}"
            ))),
        }
    }
}

#[derive(Deserialize_tuple)]
struct WireTensor {
    dtype: String,
    shape: Vec<u64>,
    data: WireTensorData,
}

/// Typed fast-path for the array-like vLLM-Omni output schema. Unused base
/// fields are skipped by serde instead of materialized as recursive Values.
#[derive(Deserialize_tuple)]
struct WireOmniOutput {
    request_id: String,
    new_token_ids: Vec<u32>,
    _new_logprobs: IgnoredAny,
    _new_prompt_logprobs_tensors: IgnoredAny,
    _pooling_output: IgnoredAny,
    finish_reason: Option<u8>,
    _stop_reason: IgnoredAny,
    _events: IgnoredAny,
    _kv_transfer_params: IgnoredAny,
    _trace_headers: IgnoredAny,
    _prefill_stats: IgnoredAny,
    _routed_experts: IgnoredAny,
    _num_nans_in_logits: u32,
    multimodal_output: Option<HashMap<String, WireTensor>>,
    _is_segment_finished: Option<bool>,
    _new_prompt_len_snapshot: Option<u64>,
}

#[derive(Deserialize_tuple)]
struct WireEngineCoreOutputs {
    _engine_index: u32,
    outputs: Vec<WireOmniOutput>,
    _scheduler_stats: IgnoredAny,
    _timestamp: f64,
    _utility_output: IgnoredAny,
    _finished_requests: IgnoredAny,
    _wave_complete: IgnoredAny,
    _start_wave: IgnoredAny,
}

#[derive(Debug)]
struct RequestMeasurement {
    generated_tokens: usize,
    tensor_deltas: usize,
    ttft: Duration,
    latency: Duration,
}

#[derive(Debug)]
struct WaveMeasurements {
    wall_time: Duration,
    requests: Vec<RequestMeasurement>,
}

struct RequestProgress {
    started_at: Instant,
    first_token_at: Option<Duration>,
    generated_tokens: usize,
    tensor_deltas: usize,
    finished_at: Option<Duration>,
}

#[derive(Clone, Debug)]
struct CorpusItem {
    utterance_id: String,
    text: String,
}

#[derive(Debug)]
struct FullRequestMeasurement {
    utterance_id: String,
    generated_tokens: usize,
    audio_samples: usize,
    sample_rate: u32,
    ttfa: Duration,
    latency: Duration,
    chunk_arrivals: Vec<Duration>,
    chunk_durations: Vec<Duration>,
    captured_audio: Vec<f32>,
}

#[derive(Debug)]
struct FullWaveMeasurements {
    seed: u64,
    wall_time: Duration,
    requests: Vec<FullRequestMeasurement>,
    frontend: FrontendMeasurements,
}

#[derive(Debug, Default)]
struct FrontendMeasurements {
    tokenization: Duration,
    request_build: Duration,
    serialization: Duration,
    codec_queue_admission: Duration,
    talker_queue_admission: Duration,
    first_talker_output_after_admission: Option<Duration>,
    first_audio_after_admission: Option<Duration>,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("easymagpie_rust_engine_poc=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn map(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

fn scalar_entry(value: Value) -> Value {
    map([
        ("tensor_data", Value::Nil),
        ("tensor_shape", Value::Nil),
        ("tensor_dtype", Value::Nil),
        ("list_data", Value::Nil),
        ("scalar_data", value),
    ])
}

fn list_entry(values: Vec<Value>) -> Value {
    map([
        ("tensor_data", Value::Nil),
        ("tensor_shape", Value::Nil),
        ("tensor_dtype", Value::Nil),
        ("list_data", Value::Array(values)),
        ("scalar_data", Value::Nil),
    ])
}

fn build_additional_information(
    args: &Args,
    request_id: &str,
    text: &str,
    final_stage_id: u64,
    token_ids: Option<RequestTokenIdSlices<'_>>,
) -> Value {
    let mut entries = vec![
        (
            Value::from("context_text"),
            scalar_entry(Value::from(args.context_text.clone())),
        ),
        (
            Value::from("text"),
            scalar_entry(Value::from(text.to_owned())),
        ),
        (
            Value::from("temperature"),
            scalar_entry(Value::from(args.audio_temperature)),
        ),
        (
            Value::from("top_k"),
            scalar_entry(Value::from(args.audio_top_k)),
        ),
        (
            Value::from("speaker_id"),
            scalar_entry(Value::from(args.speaker_id.clone())),
        ),
        (
            Value::from("global_request_id"),
            list_entry(vec![Value::from(request_id.to_owned())]),
        ),
        (
            Value::from("omni_final_stage_id"),
            scalar_entry(Value::from(final_stage_id)),
        ),
    ];
    if let Some(token_ids) = token_ids {
        entries.push((
            Value::from("context_token_ids"),
            list_entry(
                token_ids
                    .context_ids
                    .iter()
                    .copied()
                    .map(Value::from)
                    .collect(),
            ),
        ));
        entries.push((
            Value::from("text_tokens"),
            list_entry(
                token_ids
                    .text_ids
                    .iter()
                    .copied()
                    .map(Value::from)
                    .collect(),
            ),
        ));
    }
    map([("entries", Value::Map(entries))])
}

fn build_sampling_params(args: &Args) -> Value {
    let stop_ids = args
        .stop_token_id
        .map(|token| vec![Value::from(token)])
        .unwrap_or_default();
    map([
        ("temperature", Value::from(0.0_f64)),
        ("stop", Value::Array(Vec::new())),
        ("stop_token_ids", Value::Array(stop_ids.clone())),
        ("ignore_eos", Value::from(true)),
        ("max_tokens", Value::from(args.max_tokens)),
        ("detokenize", Value::from(false)),
        ("_all_stop_token_ids", Value::Array(stop_ids)),
        ("bad_words", Value::Array(Vec::new())),
        ("skip_reading_prefix_cache", Value::from(false)),
    ])
}

fn build_codec_sampling_params() -> Value {
    map([
        ("temperature", Value::from(0.0_f64)),
        ("stop", Value::Array(Vec::new())),
        ("stop_token_ids", Value::Array(Vec::new())),
        ("max_tokens", Value::from(65_536)),
        ("bad_words", Value::Array(Vec::new())),
        ("skip_reading_prefix_cache", Value::from(false)),
    ])
}

/// vLLM-Omni 0.24's OmniEngineCoreRequest is an array-like msgspec struct:
/// the 20 base EngineCoreRequest fields followed by additional_information.
fn build_engine_request(
    request_id: &str,
    prompt_len: usize,
    sampling_params: Value,
    additional_information: Value,
) -> Value {
    Value::Array(vec![
        Value::from(request_id.to_owned()),
        Value::Array(vec![Value::from(0); prompt_len]),
        Value::Nil, // mm_features
        sampling_params,
        Value::Nil,       // pooling_params
        Value::from(0.0), // arrival_time
        Value::Nil,       // lora_request
        Value::Nil,       // cache_salt
        Value::Nil,       // data_parallel_rank
        Value::Nil,       // prompt_embeds
        Value::Nil,       // prompt_is_token_ids
        Value::from(0),   // client_index
        Value::from(0),   // current_wave
        Value::from(0),   // priority
        Value::Nil,       // trace_headers
        Value::from(false),
        Value::from(request_id.to_owned()), // external_req_id
        Value::Nil,                         // reasoning_ended
        Value::Nil,                         // reasoning_parser_kwargs
        Value::from(false),                 // abort_immediately
        additional_information,
    ])
}

fn build_request(
    args: &Args,
    request_id: &str,
    text: &str,
    tokenizer: Option<&RustTextTokenizer>,
) -> Result<Value> {
    let token_ids = tokenizer
        .map(|tokenizer| tokenizer.encode_request(text))
        .transpose()?;
    Ok(build_engine_request(
        request_id,
        args.prompt_len,
        build_sampling_params(args),
        build_additional_information(
            args,
            request_id,
            text,
            u64::from(args.full_pipeline),
            token_ids.as_ref().map(RequestTokenIds::as_slices),
        ),
    ))
}

fn build_request_with_token_slices(
    args: &Args,
    request_id: &str,
    text: &str,
    token_ids: Option<RequestTokenIdSlices<'_>>,
) -> Value {
    build_engine_request(
        request_id,
        args.prompt_len,
        build_sampling_params(args),
        build_additional_information(
            args,
            request_id,
            text,
            u64::from(args.full_pipeline),
            token_ids,
        ),
    )
}

fn build_codec_request(args: &Args, request_id: &str, text: &str) -> Value {
    build_engine_request(
        request_id,
        1,
        build_codec_sampling_params(),
        build_additional_information(args, request_id, text, 1, None),
    )
}

fn encode(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, value).context("encode MessagePack")?;
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<Value> {
    rmpv::decode::read_value(&mut Cursor::new(bytes)).context("decode MessagePack")
}

fn map_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    let Value::Map(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find_map(|(candidate, value)| (candidate.as_str() == Some(key)).then_some(value))
}

fn status(value: &Value) -> Option<&str> {
    map_get(value, "status").and_then(Value::as_str)
}

async fn bind_data_socket_pair(host: &str) -> Result<(String, RouterSocket, String, PullSocket)> {
    let mut input_socket = RouterSocket::new();
    let input_address = input_socket
        .bind(&format!("tcp://{host}:0"))
        .await
        .context("bind engine input ROUTER")?
        .to_string();

    let mut output_socket = PullSocket::new();
    let output_address = output_socket
        .bind(&format!("tcp://{host}:0"))
        .await
        .context("bind engine output PULL")?
        .to_string();

    Ok((input_address, input_socket, output_address, output_socket))
}

fn init_message(input_address: &str, output_address: &str) -> Value {
    map([
        (
            "addresses",
            map([
                (
                    "inputs",
                    Value::Array(vec![Value::from(input_address.to_owned())]),
                ),
                (
                    "outputs",
                    Value::Array(vec![Value::from(output_address.to_owned())]),
                ),
                ("coordinator_input", Value::Nil),
                ("coordinator_output", Value::Nil),
                ("frontend_stats_publish_address", Value::Nil),
            ]),
        ),
        ("parallel_config", Value::Map(Vec::new())),
    ])
}

fn two_frames(message: ZmqMessage, context: &str) -> Result<(Bytes, Bytes)> {
    let frames = message.into_vec();
    if frames.len() != 2 {
        bail!(
            "{context}: expected 2 ZMQ frames, received {}",
            frames.len()
        );
    }
    Ok((frames[0].clone(), frames[1].clone()))
}

async fn connect_engine(
    args: &Args,
    handshake_address: &str,
    engine_name: &str,
) -> Result<ConnectedEngine> {
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let (input_address, mut input_socket, output_address, output_socket) =
        bind_data_socket_pair(&args.advertised_host).await?;

    let mut handshake = RouterSocket::new();
    handshake
        .bind(handshake_address)
        .await
        .with_context(|| format!("bind {engine_name} handshake {handshake_address}"))?;

    info!(
        %engine_name,
        handshake = %handshake_address,
        "waiting for vLLM-Omni StageEngineCoreProc"
    );

    let hello = timeout(ready_timeout, handshake.recv())
        .await
        .context("timed out waiting for engine HELLO")?
        .context("receive engine HELLO")?;
    let (identity, hello_payload) = two_frames(hello, "HELLO")?;
    let hello_value = decode(&hello_payload)?;
    if status(&hello_value) != Some("HELLO") {
        bail!("expected HELLO, received {hello_value:?}");
    }

    let init_payload = encode(&init_message(&input_address, &output_address))?;
    let init = ZmqMessage::try_from(vec![identity.clone(), Bytes::from(init_payload)])
        .map_err(|error| anyhow!("build INIT ZMQ message: {error}"))?;
    handshake.send(init).await.context("send engine INIT")?;

    let ready = timeout(ready_timeout, handshake.recv())
        .await
        .context("timed out waiting for engine READY")?
        .context("receive engine READY")?;
    let (ready_identity, ready_payload) = two_frames(ready, "READY")?;
    if ready_identity != identity {
        bail!("engine identity changed between HELLO and READY");
    }
    let ready_value = decode(&ready_payload)?;
    if status(&ready_value) != Some("READY") {
        bail!("expected READY, received {ready_value:?}");
    }

    let registration = timeout(ready_timeout, input_socket.recv())
        .await
        .context("timed out waiting for engine input registration")?
        .context("receive engine input registration")?;
    let (registered_identity, registration_payload) = two_frames(registration, "registration")?;
    if registered_identity != identity {
        bail!("engine identity changed during input registration");
    }
    debug!(registration = ?decode(&registration_payload)?, "engine registered");

    let (input_send, _) = input_socket.split();
    info!(%engine_name, %input_address, %output_address, "engine transport ready");
    Ok(ConnectedEngine {
        identity,
        input_send,
        output_socket,
    })
}

async fn send_to_engine(
    input: &mut RouterSendHalf,
    identity: &Bytes,
    request_type: &[u8],
    payload: Vec<u8>,
) -> Result<()> {
    let message = ZmqMessage::try_from(vec![
        identity.clone(),
        Bytes::copy_from_slice(request_type),
        Bytes::from(payload),
    ])
    .map_err(|error| anyhow!("build engine input message: {error}"))?;
    input.send(message).await.context("send engine input")
}

#[cfg(test)]
fn value_array(value: &Value) -> Option<&[Value]> {
    let Value::Array(values) = value else {
        return None;
    };
    Some(values)
}

#[cfg(test)]
fn tensor_from_value<'a>(
    key: String,
    value: &'a Value,
    frames: &'a [Bytes],
) -> Option<TensorDelta<'a>> {
    let tuple = value_array(value)?;
    if tuple.len() != 3 {
        return None;
    }
    let dtype = tuple[0].as_str()?.to_owned();
    let shape = value_array(&tuple[1])?
        .iter()
        .map(Value::as_u64)
        .collect::<Option<Vec<_>>>()?;
    let bytes = match &tuple[2] {
        Value::Integer(index) => frames.get(index.as_u64()? as usize)?.as_ref(),
        Value::Ext(3, bytes) => bytes.as_slice(),
        _ => return None,
    };
    Some(TensorDelta {
        key,
        dtype,
        shape,
        bytes,
    })
}

fn sanitize_filename(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn save_tensor(output_dir: &Path, sequence: usize, tensor: &TensorDelta<'_>) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("create output directory {}", output_dir.display()))?;
    let stem = format!("{sequence:05}_{}", sanitize_filename(&tensor.key));
    fs::write(output_dir.join(format!("{stem}.bin")), tensor.bytes)
        .with_context(|| format!("write tensor {stem}"))?;
    fs::write(
        output_dir.join(format!("{stem}.txt")),
        format!(
            "key={}\ndtype={}\nshape={:?}\nbytes={}\n",
            tensor.key,
            tensor.dtype,
            tensor.shape,
            tensor.bytes.len()
        ),
    )
    .with_context(|| format!("write tensor metadata {stem}"))?;
    Ok(())
}

fn is_audio_tensor_key(key: &str) -> bool {
    key.contains("model_outputs") || key.ends_with("audio")
}

fn tensor_element_count(tensor: &TensorDelta<'_>) -> Option<usize> {
    tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
}

fn tensor_scalar_u32_parts(dtype: &str, bytes: &[u8]) -> Option<u32> {
    match dtype {
        "int32" if bytes.len() >= 4 => Some(i32::from_le_bytes(bytes[..4].try_into().ok()?) as u32),
        "int64" if bytes.len() >= 8 => Some(i64::from_le_bytes(bytes[..8].try_into().ok()?) as u32),
        _ => None,
    }
}

fn process_output_message(
    message: ZmqMessage,
    output_dir: Option<&Path>,
    tensor_sequence: &mut usize,
    collect_tensor_data: bool,
) -> Result<Vec<OutputEvent>> {
    let frames = message.into_vec();
    let first = frames.first().context("engine output had no frames")?;
    if first.as_ref() == ENGINE_CORE_DEAD {
        bail!("engine sent ENGINE_CORE_DEAD");
    }
    let batch: WireEngineCoreOutputs =
        rmp_serde::from_slice(first).context("decode typed EngineCoreOutputs MessagePack")?;

    let mut events = Vec::new();
    for output in batch.outputs {
        let request_id = output.request_id;
        let new_tokens = output.new_token_ids.len();
        let mut tensor_deltas = 0;
        let mut audio_sample_counts = Vec::new();
        let mut sample_rate = None;
        let mut owned_tensors = Vec::new();
        if let Some(multimodal) = &output.multimodal_output {
            for (key, wire_tensor) in multimodal {
                let bytes = match &wire_tensor.data {
                    WireTensorData::AuxIndex(index) => frames
                        .get(*index)
                        .with_context(|| format!("tensor {key} referenced missing frame {index}"))?
                        .as_ref(),
                    WireTensorData::Raw(bytes) => bytes.as_slice(),
                };
                let tensor = TensorDelta {
                    key: key.clone(),
                    dtype: wire_tensor.dtype.clone(),
                    shape: wire_tensor.shape.clone(),
                    bytes,
                };
                debug!(
                    request_id = %request_id,
                    key = %tensor.key,
                    dtype = %tensor.dtype,
                    shape = ?tensor.shape,
                    bytes = tensor.bytes.len(),
                    "received multimodal tensor delta"
                );
                if let Some(output_dir) = output_dir {
                    save_tensor(output_dir, *tensor_sequence, &tensor)?;
                }
                if is_audio_tensor_key(&tensor.key) && tensor.dtype == "float32" {
                    if let Some(count) = tensor_element_count(&tensor).filter(|count| *count > 0) {
                        audio_sample_counts.push(count);
                    }
                } else if tensor.key.ends_with("sr") || tensor.key.contains("sr[") {
                    sample_rate = tensor_scalar_u32_parts(&tensor.dtype, tensor.bytes);
                }
                if collect_tensor_data {
                    owned_tensors.push(OwnedTensor {
                        key: tensor.key.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        bytes: tensor.bytes.to_vec(),
                    });
                }
                *tensor_sequence += 1;
                tensor_deltas += 1;
            }
        }
        let finished = output.finish_reason.is_some();
        events.push(OutputEvent {
            request_id,
            new_tokens,
            tensor_deltas,
            finished,
            audio_sample_counts,
            sample_rate,
            tensors: owned_tensors,
        });
    }
    Ok(events)
}

async fn abort_request(engine: &mut ConnectedEngine, request_id: &str) -> Result<()> {
    let payload = encode(&Value::Array(vec![Value::from(request_id.to_owned())]))?;
    send_to_engine(
        &mut engine.input_send,
        &engine.identity,
        REQUEST_TYPE_ABORT,
        payload,
    )
    .await
}

async fn run_wave(
    engine: &mut ConnectedEngine,
    args: &Args,
    tokenizer: Option<&RustTextTokenizer>,
    wave_index: usize,
    tensor_sequence: &mut usize,
) -> Result<WaveMeasurements> {
    let wave_started_at = Instant::now();
    let unix_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let mut progress = HashMap::with_capacity(args.batch_size);

    for request_index in 0..args.batch_size {
        let request_id = format!("rust-easymagpie-{unix_nanos}-w{wave_index}-r{request_index}");
        let request = build_request(args, &request_id, &args.text, tokenizer)?;
        let payload = encode(&request)?;
        let started_at = Instant::now();
        send_to_engine(
            &mut engine.input_send,
            &engine.identity,
            REQUEST_TYPE_ADD,
            payload,
        )
        .await?;
        progress.insert(
            request_id,
            RequestProgress {
                started_at,
                first_token_at: None,
                generated_tokens: 0,
                tensor_deltas: 0,
                finished_at: None,
            },
        );
    }

    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let result = timeout(output_timeout, async {
        let mut completed = 0;
        while completed < args.batch_size {
            let message = engine
                .output_socket
                .recv()
                .await
                .context("receive EngineCore output")?;
            for event in
                process_output_message(message, args.output_dir.as_deref(), tensor_sequence, false)?
            {
                let Some(request) = progress.get_mut(&event.request_id) else {
                    continue;
                };
                let elapsed = request.started_at.elapsed();
                if event.new_tokens > 0 && request.first_token_at.is_none() {
                    request.first_token_at = Some(elapsed);
                }
                request.generated_tokens += event.new_tokens;
                request.tensor_deltas += event.tensor_deltas;
                if event.finished && request.finished_at.is_none() {
                    request.finished_at = Some(elapsed);
                    completed += 1;
                }
            }
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;

    match result {
        Ok(result) => result?,
        Err(_) => {
            for (request_id, request) in &progress {
                if request.finished_at.is_none() {
                    let _ = abort_request(engine, request_id).await;
                }
            }
            bail!(
                "wave {wave_index} timed out after {} seconds; outstanding requests were aborted",
                args.output_timeout_secs
            );
        }
    }

    let requests = progress
        .into_values()
        .map(|request| {
            let latency = request
                .finished_at
                .context("request completed without a finish timestamp")?;
            Ok(RequestMeasurement {
                generated_tokens: request.generated_tokens,
                tensor_deltas: request.tensor_deltas,
                ttft: request.first_token_at.unwrap_or(latency),
                latency,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(WaveMeasurements {
        wall_time: wave_started_at.elapsed(),
        requests,
    })
}

fn percentile_ms(values: impl Iterator<Item = Duration>, percentile: f64) -> f64 {
    let mut values = values
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn tensor_f32_values(tensor: &OwnedTensor) -> Result<Vec<f32>> {
    if tensor.dtype != "float32" {
        bail!(
            "unsupported audio tensor dtype {} for {}",
            tensor.dtype,
            tensor.key
        );
    }
    if !tensor.bytes.len().is_multiple_of(4) {
        bail!(
            "float32 tensor {} has {} bytes",
            tensor.key,
            tensor.bytes.len()
        );
    }
    let elements = tensor.bytes.len() / 4;
    let expected = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .context("audio tensor shape overflow")?;
    if expected != elements {
        bail!(
            "audio tensor {} shape {:?} implies {expected} values but carries {elements}",
            tensor.key,
            tensor.shape
        );
    }
    Ok(tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn tensor_scalar_u32(tensor: &OwnedTensor) -> Option<u32> {
    tensor_scalar_u32_parts(&tensor.dtype, &tensor.bytes)
}

fn write_pcm16_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let data_size = samples
        .len()
        .checked_mul(2)
        .and_then(|size| u32::try_from(size).ok())
        .context("WAV data exceeds the RIFF size limit")?;
    let riff_size = 36_u32
        .checked_add(data_size)
        .context("WAV RIFF size overflow")?;
    let mut wav = Vec::with_capacity(data_size as usize + 44);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        wav.extend_from_slice(&pcm.to_le_bytes());
    }
    fs::write(path, wav).with_context(|| format!("write WAV {}", path.display()))
}

async fn send_add_request_bytes(engine: &mut ConnectedEngine, request: Vec<u8>) -> Result<()> {
    send_to_engine(
        &mut engine.input_send,
        &engine.identity,
        REQUEST_TYPE_ADD,
        request,
    )
    .await
}

/// CPython's MT19937 integer seeding and 53-bit `random()` implementation.
/// This keeps BS16 corpus selection identical to benchmark_server.py's
/// `random.seed(seed); random.choices(items, k=16)`.
struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u64) -> Self {
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.state[0] = 19_650_218;
        for index in 1..624 {
            random.state[index] = 1_812_433_253_u32
                .wrapping_mul(random.state[index - 1] ^ (random.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        let mut keys = vec![seed as u32];
        if seed > u32::MAX as u64 {
            keys.push((seed >> 32) as u32);
        }
        let mut state_index = 1;
        let mut key_index = 0;
        for _ in 0..624.max(keys.len()) {
            random.state[state_index] = (random.state[state_index]
                ^ (random.state[state_index - 1] ^ (random.state[state_index - 1] >> 30))
                    .wrapping_mul(1_664_525))
            .wrapping_add(keys[key_index])
            .wrapping_add(key_index as u32);
            state_index += 1;
            key_index += 1;
            if state_index >= 624 {
                random.state[0] = random.state[623];
                state_index = 1;
            }
            if key_index >= keys.len() {
                key_index = 0;
            }
        }
        for _ in 0..623 {
            random.state[state_index] = (random.state[state_index]
                ^ (random.state[state_index - 1] ^ (random.state[state_index - 1] >> 30))
                    .wrapping_mul(1_566_083_941))
            .wrapping_sub(state_index as u32);
            state_index += 1;
            if state_index >= 624 {
                random.state[0] = random.state[623];
                state_index = 1;
            }
        }
        random.state[0] = 0x8000_0000;
        random
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                let mut next = self.state[(index + 397) % 624] ^ (value >> 1);
                if value & 1 != 0 {
                    next ^= 0x9908_b0df;
                }
                self.state[index] = next;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn random(&mut self) -> f64 {
        let upper = (self.next_u32() >> 5) as u64;
        let lower = (self.next_u32() >> 6) as u64;
        (upper * 67_108_864 + lower) as f64 / 9_007_199_254_740_992.0
    }
}

fn load_corpus(args: &Args) -> Result<Vec<CorpusItem>> {
    let Some(path) = &args.text_file else {
        return Ok(vec![CorpusItem {
            utterance_id: "text".to_owned(),
            text: args.text.clone(),
        }]);
    };
    let contents =
        fs::read_to_string(path).with_context(|| format!("read corpus {}", path.display()))?;
    let mut items = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (utterance_id, text) = line.split_once('\t').with_context(|| {
            format!(
                "{}:{} must contain '<utterance-id>\\t<text>'",
                path.display(),
                line_index + 1
            )
        })?;
        if utterance_id.trim().is_empty() || text.trim().is_empty() {
            bail!(
                "{}:{} contains an empty utterance id or text",
                path.display(),
                line_index + 1
            );
        }
        items.push(CorpusItem {
            utterance_id: utterance_id.trim().to_owned(),
            text: text.trim().to_owned(),
        });
    }
    if items.is_empty() {
        bail!("corpus {} contains no usable lines", path.display());
    }
    Ok(items)
}

fn select_corpus_wave(items: &[CorpusItem], batch_size: usize, seed: u64) -> Vec<CorpusItem> {
    let mut random = PythonRandom::new(seed);
    (0..batch_size)
        .map(|_| {
            let index = (random.random() * items.len() as f64) as usize;
            items[index.min(items.len() - 1)].clone()
        })
        .collect()
}

struct FullRequestProgress {
    utterance_id: String,
    started_at: Option<Instant>,
    talker_finished: bool,
    codec_finished: bool,
    generated_tokens: usize,
    audio_samples: usize,
    sample_rate: u32,
    first_audio_at: Option<Duration>,
    finished_at: Option<Duration>,
    chunk_arrivals: Vec<Duration>,
    chunk_durations: Vec<Duration>,
    captured_audio: Vec<f32>,
}

async fn run_full_wave(
    talker: &mut ConnectedEngine,
    codec: &mut ConnectedEngine,
    args: &Args,
    tokenizer: Option<&RustTextTokenizer>,
    items: &[CorpusItem],
    wave_index: usize,
    seed: u64,
    capture_audio: bool,
    tensor_sequence: &mut usize,
) -> Result<FullWaveMeasurements> {
    let unix_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let selected = select_corpus_wave(items, args.batch_size, seed);
    let mut request_ids = Vec::with_capacity(args.batch_size);
    let mut progress = HashMap::with_capacity(args.batch_size);

    for (request_index, item) in selected.iter().enumerate() {
        let request_id =
            format!("rust-easymagpie-full-{unix_nanos}-w{wave_index}-r{request_index:03}");
        request_ids.push(request_id.clone());
        progress.insert(
            request_id,
            FullRequestProgress {
                utterance_id: item.utterance_id.clone(),
                started_at: None,
                talker_finished: false,
                codec_finished: false,
                generated_tokens: 0,
                audio_samples: 0,
                sample_rate: args.sample_rate,
                first_audio_at: None,
                finished_at: None,
                chunk_arrivals: Vec::new(),
                chunk_durations: Vec::new(),
                captured_audio: Vec::new(),
            },
        );
    }

    // Tokenize the whole cohort in one call. Hugging Face tokenizers can
    // parallelize encode_batch internally, avoiding 16 serial FFI/API calls.
    let tokenization_started_at = Instant::now();
    let target_token_ids = tokenizer
        .map(|tokenizer| {
            let texts = selected
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>();
            tokenizer.encode_batch(&texts)
        })
        .transpose()?;
    let tokenization = tokenization_started_at.elapsed();

    // Build Stage-0 and Stage-1 request values in parallel. This is pure CPU
    // work and does not touch either ZMQ socket.
    let request_build_started_at = Instant::now();
    let request_values = request_ids
        .par_iter()
        .zip(selected.par_iter())
        .enumerate()
        .map(|(index, (request_id, item))| {
            let token_slices =
                tokenizer
                    .zip(target_token_ids.as_ref())
                    .map(|(tokenizer, target_token_ids)| RequestTokenIdSlices {
                        context_ids: &tokenizer.context_ids,
                        text_ids: &target_token_ids[index],
                    });
            (
                build_codec_request(args, request_id, &item.text),
                build_request_with_token_slices(args, request_id, &item.text, token_slices),
            )
        })
        .collect::<Vec<_>>();
    let request_build = request_build_started_at.elapsed();

    // MessagePack serialization is independent for every request, so perform
    // it in parallel before entering the async socket-admission path.
    let serialization_started_at = Instant::now();
    let serialized_requests = request_values
        .par_iter()
        .map(|(codec_request, talker_request)| {
            Ok((encode(codec_request)?, encode(talker_request)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let serialization = serialization_started_at.elapsed();
    let (codec_requests, talker_requests): (Vec<_>, Vec<_>) =
        serialized_requests.into_iter().unzip();

    // Stage 1 must be polling the connector before Stage 0 publishes the
    // first acoustic packet. Pre-submit the entire cohort downstream.
    let codec_admission_started_at = Instant::now();
    for codec_request in codec_requests {
        send_add_request_bytes(codec, codec_request).await?;
    }
    let codec_queue_admission = codec_admission_started_at.elapsed();

    let wave_started_at = Instant::now();
    let talker_admission_started_at = Instant::now();
    for (request_id, talker_request) in request_ids.iter().zip(talker_requests) {
        let started_at = Instant::now();
        progress
            .get_mut(request_id)
            .expect("request progress exists")
            .started_at = Some(started_at);
        send_add_request_bytes(talker, talker_request).await?;
    }
    let talker_queue_admission = talker_admission_started_at.elapsed();
    let admitted_at = Instant::now();
    let mut first_talker_output_after_admission = None;
    let mut first_audio_after_admission = None;

    let mut completed = 0;
    let mut talker_completed = 0;
    let mut codec_completed = 0;
    let mut last_completion_at = Duration::ZERO;
    let result = timeout(Duration::from_secs(args.output_timeout_secs), async {
        while completed < args.batch_size {
            tokio::select! {
                message = talker.output_socket.recv(), if talker_completed < args.batch_size => {
                    let received_at = Instant::now();
                    let message = message.context("receive talker output")?;
                    let events = process_output_message(
                        message,
                        args.output_dir.as_deref(),
                        tensor_sequence,
                        false,
                    )?;
                    if first_talker_output_after_admission.is_none()
                        && events.iter().any(|event| progress.contains_key(&event.request_id))
                    {
                        first_talker_output_after_admission =
                            Some(received_at.duration_since(admitted_at));
                    }
                    for event in events {
                        let Some(request) = progress.get_mut(&event.request_id) else {
                            continue;
                        };
                        request.generated_tokens += event.new_tokens;
                        if event.finished && !request.talker_finished {
                            request.talker_finished = true;
                            talker_completed += 1;
                        }
                        if request.talker_finished
                            && request.codec_finished
                            && request.finished_at.is_none()
                        {
                            let started_at = request.started_at.context("talker request was not submitted")?;
                            request.finished_at = Some(received_at.duration_since(started_at));
                            last_completion_at = received_at.duration_since(wave_started_at);
                            completed += 1;
                        }
                    }
                }
                message = codec.output_socket.recv(), if codec_completed < args.batch_size => {
                    // Timestamp immediately on socket receipt. Tensor parsing, optional
                    // PCM conversion, and WAV I/O are outside TTFA/latency.
                    let received_at = Instant::now();
                    let message = message.context("receive codec output")?;
                    let events = process_output_message(
                        message,
                        args.output_dir.as_deref(),
                        tensor_sequence,
                        capture_audio,
                    )?;
                    if first_audio_after_admission.is_none()
                        && events.iter().any(|event| {
                            progress.contains_key(&event.request_id)
                                && !event.audio_sample_counts.is_empty()
                        })
                    {
                        first_audio_after_admission = Some(received_at.duration_since(admitted_at));
                    }
                    for event in events {
                        let Some(request) = progress.get_mut(&event.request_id) else {
                            continue;
                        };
                        let started_at = request.started_at.context("talker request was not submitted")?;
                        let arrival = received_at.duration_since(started_at);
                        if let Some(sample_rate) = event.sample_rate {
                            request.sample_rate = sample_rate;
                        }
                        for sample_count in event.audio_sample_counts {
                            if request.first_audio_at.is_none() {
                                request.first_audio_at = Some(arrival);
                            }
                            request.audio_samples += sample_count;
                            request.chunk_arrivals.push(arrival);
                            request.chunk_durations.push(Duration::from_secs_f64(
                                sample_count as f64 / request.sample_rate as f64,
                            ));
                        }
                        if capture_audio {
                            for tensor in event.tensors {
                                if is_audio_tensor_key(&tensor.key) {
                                    request.captured_audio.extend(tensor_f32_values(&tensor)?);
                                } else if tensor.key.ends_with("sr") || tensor.key.contains("sr[") {
                                    if let Some(sample_rate) = tensor_scalar_u32(&tensor) {
                                        request.sample_rate = sample_rate;
                                    }
                                }
                            }
                        }
                        if event.finished && !request.codec_finished {
                            request.codec_finished = true;
                            codec_completed += 1;
                        }
                        if request.talker_finished
                            && request.codec_finished
                            && request.finished_at.is_none()
                        {
                            request.finished_at = Some(arrival);
                            last_completion_at = received_at.duration_since(wave_started_at);
                            completed += 1;
                        }
                    }
                }
            }
            // zeromq's receive future can remain immediately ready while a
            // multipart burst is being drained. Cooperatively yield so the
            // Tokio timer and socket-reactor tasks cannot be starved across
            // long, back-to-back cohorts.
            tokio::task::yield_now().await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;

    match result {
        Ok(result) => result?,
        Err(_) => {
            for request_id in &request_ids {
                if progress
                    .get(request_id)
                    .is_some_and(|request| request.finished_at.is_none())
                {
                    let _ = abort_request(talker, request_id).await;
                    let _ = abort_request(codec, request_id).await;
                }
            }
            bail!(
                "full-pipeline wave {wave_index} timed out after {} seconds; outstanding requests were aborted",
                args.output_timeout_secs
            );
        }
    }

    let mut requests = Vec::with_capacity(args.batch_size);
    for request_id in request_ids {
        let request = progress
            .remove(&request_id)
            .context("completed request disappeared from progress map")?;
        if request.audio_samples == 0 {
            bail!("{request_id} completed without returning audio samples");
        }
        requests.push(FullRequestMeasurement {
            utterance_id: request.utterance_id,
            generated_tokens: request.generated_tokens,
            audio_samples: request.audio_samples,
            sample_rate: request.sample_rate,
            ttfa: request
                .first_audio_at
                .context("codec completed without a first-audio timestamp")?,
            latency: request
                .finished_at
                .context("request completed without a finish timestamp")?,
            chunk_arrivals: request.chunk_arrivals,
            chunk_durations: request.chunk_durations,
            captured_audio: request.captured_audio,
        });
    }
    Ok(FullWaveMeasurements {
        seed,
        wall_time: last_completion_at,
        requests,
        frontend: FrontendMeasurements {
            tokenization,
            request_build,
            serialization,
            codec_queue_admission,
            talker_queue_admission,
            first_talker_output_after_admission,
            first_audio_after_admission,
        },
    })
}

fn percentile_f64(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn mean_duration_ms(values: impl Iterator<Item = Duration>) -> f64 {
    let values = values
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn tokenize_texts(tokenizer: &RustTextTokenizer, texts: &[&str], use_batch: bool) -> Result<usize> {
    if use_batch {
        return Ok(tokenizer.encode_batch(texts)?.iter().map(Vec::len).sum());
    }
    texts.iter().try_fold(0_usize, |total, text| {
        Ok(total + tokenizer.encode_request(text)?.text_ids.len())
    })
}

fn playback_metrics(requests: &[&FullRequestMeasurement]) -> (usize, usize, usize, Vec<f64>) {
    let mut chunks = 0;
    let mut underruns = 0;
    let mut deadline_misses = 0;
    let mut gaps_ms = Vec::new();
    for request in requests {
        chunks += request.chunk_arrivals.len();
        let Some(first_arrival) = request.chunk_arrivals.first() else {
            continue;
        };
        let mut playback_end =
            first_arrival.as_secs_f64() + request.chunk_durations[0].as_secs_f64();
        for index in 1..request.chunk_arrivals.len() {
            let arrival = request.chunk_arrivals[index].as_secs_f64();
            let previous_arrival = request.chunk_arrivals[index - 1].as_secs_f64();
            let previous_audio = request.chunk_durations[index - 1].as_secs_f64();
            let gap = arrival - previous_arrival;
            gaps_ms.push(gap * 1000.0);
            if gap > previous_audio {
                deadline_misses += 1;
            }
            if arrival > playback_end {
                underruns += 1;
                playback_end = arrival;
            }
            playback_end += request.chunk_durations[index].as_secs_f64();
        }
    }
    (chunks, underruns, deadline_misses, gaps_ms)
}

async fn run_full_pipeline(args: &Args, tokenizer: Option<&RustTextTokenizer>) -> Result<()> {
    let corpus = load_corpus(args)?;
    let mut talker = connect_engine(args, &args.handshake_address, "talker").await?;
    let mut codec = connect_engine(args, &args.codec_handshake_address, "codec").await?;
    let mut tensor_sequence = 0;

    for warmup_index in 0..args.warmup_rounds {
        let seed = args
            .seed
            .wrapping_sub((args.warmup_rounds - warmup_index) as u64);
        info!(
            round = warmup_index + 1,
            batch_size = args.batch_size,
            seed,
            "running full-pipeline warmup wave"
        );
        run_full_wave(
            &mut talker,
            &mut codec,
            args,
            tokenizer,
            &corpus,
            warmup_index,
            seed,
            false,
            &mut tensor_sequence,
        )
        .await?;
    }

    let capture_audio = args.batch_size == 1 && args.rounds == 1;
    let mut waves = Vec::with_capacity(args.rounds);
    for round_index in 0..args.rounds {
        let seed = args.seed.wrapping_add(round_index as u64);
        info!(
            round = round_index + 1,
            batch_size = args.batch_size,
            seed,
            "running full-pipeline measured wave"
        );
        waves.push(
            run_full_wave(
                &mut talker,
                &mut codec,
                args,
                tokenizer,
                &corpus,
                args.warmup_rounds + round_index,
                seed,
                capture_audio,
                &mut tensor_sequence,
            )
            .await?,
        );
    }

    if capture_audio {
        let request = &waves[0].requests[0];
        if request.captured_audio.is_empty() {
            bail!("audio capture was requested but the codec returned no PCM tensor");
        }
        write_pcm16_wav(
            &args.wav_output,
            &request.captured_audio,
            request.sample_rate,
        )?;
        println!("wav_output={}", args.wav_output.display());
    }

    let wall_seconds = waves
        .iter()
        .map(|wave| wave.wall_time.as_secs_f64())
        .sum::<f64>();
    let requests = waves
        .iter()
        .flat_map(|wave| wave.requests.iter())
        .collect::<Vec<_>>();
    let audio_seconds = requests
        .iter()
        .map(|request| request.audio_samples as f64 / request.sample_rate as f64)
        .sum::<f64>();
    let mut wave_rtfx = Vec::with_capacity(waves.len());
    for (index, wave) in waves.iter().enumerate() {
        let wave_audio_seconds = wave
            .requests
            .iter()
            .map(|request| request.audio_samples as f64 / request.sample_rate as f64)
            .sum::<f64>();
        let rtfx = wave_audio_seconds / wave.wall_time.as_secs_f64();
        wave_rtfx.push(rtfx);
        println!(
            "wave={} seed={} wall_seconds={:.6} audio_seconds={:.3} rtfx={:.3}",
            index + 1,
            wave.seed,
            wave.wall_time.as_secs_f64(),
            wave_audio_seconds,
            rtfx,
        );
        println!(
            "wave_{}_utterances={}",
            index + 1,
            wave.requests
                .iter()
                .map(|request| request.utterance_id.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "wave_{}_frontend_ms=tokenization:{:.3},request_build:{:.3},serialization:{:.3},codec_admission:{:.3},talker_admission:{:.3},first_talker_output:{:.3},first_audio:{:.3}",
            index + 1,
            wave.frontend.tokenization.as_secs_f64() * 1000.0,
            wave.frontend.request_build.as_secs_f64() * 1000.0,
            wave.frontend.serialization.as_secs_f64() * 1000.0,
            wave.frontend.codec_queue_admission.as_secs_f64() * 1000.0,
            wave.frontend.talker_queue_admission.as_secs_f64() * 1000.0,
            wave.frontend
                .first_talker_output_after_admission
                .unwrap_or_default()
                .as_secs_f64()
                * 1000.0,
            wave.frontend
                .first_audio_after_admission
                .unwrap_or_default()
                .as_secs_f64()
                * 1000.0,
        );
    }
    let wave_rtfx_mean = wave_rtfx.iter().sum::<f64>() / wave_rtfx.len() as f64;
    let wave_rtfx_median = percentile_f64(&mut wave_rtfx, 0.5);
    let ttfa_mean_ms = requests
        .iter()
        .map(|request| request.ttfa.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / requests.len() as f64;
    let latency_mean_ms = requests
        .iter()
        .map(|request| request.latency.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / requests.len() as f64;
    let (chunks, underruns, deadline_misses, mut gaps_ms) = playback_metrics(&requests);
    let itl_mean_ms = gaps_ms.iter().sum::<f64>() / gaps_ms.len().max(1) as f64;
    let itl_p95_ms = percentile_f64(&mut gaps_ms, 0.95);

    println!("batch_size={}", args.batch_size);
    println!("warmup_rounds={}", args.warmup_rounds);
    println!("rounds={}", args.rounds);
    println!("requests={}", requests.len());
    println!(
        "generated_tokens={}",
        requests
            .iter()
            .map(|request| request.generated_tokens)
            .sum::<usize>()
    );
    println!("wall_seconds={wall_seconds:.6}");
    println!("audio_seconds={audio_seconds:.3}");
    println!(
        "request_throughput_per_second={:.3}",
        requests.len() as f64 / wall_seconds
    );
    println!("aggregate_rtfx={:.3}", audio_seconds / wall_seconds);
    println!("wave_rtfx_mean={wave_rtfx_mean:.3}");
    println!("wave_rtfx_median={wave_rtfx_median:.3}");
    println!("ttfa_mean_ms={ttfa_mean_ms:.3}");
    println!(
        "ttfa_p95_ms={:.3}",
        percentile_ms(requests.iter().map(|request| request.ttfa), 0.95)
    );
    println!("latency_mean_ms={latency_mean_ms:.3}");
    println!(
        "latency_p95_ms={:.3}",
        percentile_ms(requests.iter().map(|request| request.latency), 0.95)
    );
    println!("itl_mean_ms={itl_mean_ms:.3}");
    println!("itl_p95_ms={itl_p95_ms:.3}");
    println!(
        "frontend_tokenization_mean_ms={:.3}",
        mean_duration_ms(waves.iter().map(|wave| wave.frontend.tokenization))
    );
    println!(
        "frontend_request_build_mean_ms={:.3}",
        mean_duration_ms(waves.iter().map(|wave| wave.frontend.request_build))
    );
    println!(
        "frontend_serialization_mean_ms={:.3}",
        mean_duration_ms(waves.iter().map(|wave| wave.frontend.serialization))
    );
    println!(
        "codec_queue_admission_mean_ms={:.3}",
        mean_duration_ms(waves.iter().map(|wave| wave.frontend.codec_queue_admission),)
    );
    println!(
        "talker_queue_admission_mean_ms={:.3}",
        mean_duration_ms(
            waves
                .iter()
                .map(|wave| wave.frontend.talker_queue_admission),
        )
    );
    println!(
        "first_talker_output_after_admission_mean_ms={:.3}",
        mean_duration_ms(
            waves
                .iter()
                .filter_map(|wave| { wave.frontend.first_talker_output_after_admission })
        )
    );
    println!(
        "first_audio_after_admission_mean_ms={:.3}",
        mean_duration_ms(
            waves
                .iter()
                .filter_map(|wave| wave.frontend.first_audio_after_admission),
        )
    );
    println!("audio_chunks={chunks}");
    println!("playback_deadline_misses={deadline_misses}");
    println!("playback_underruns={underruns}");
    println!(
        "playback_underrun_percent={:.3}",
        if chunks == 0 {
            0.0
        } else {
            underruns as f64 * 100.0 / chunks as f64
        }
    );
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than zero");
    }
    if args.rounds == 0 {
        bail!("--rounds must be greater than zero");
    }
    if args.frame_stacking_factor == 0 || args.codec_frame_rate <= 0.0 {
        bail!("audio-duration parameters must be greater than zero");
    }
    if args.tokenize_iterations == 0 {
        bail!("--tokenize-iterations must be greater than zero");
    }
    if args.tokenize_batch_size == Some(0) {
        bail!("--tokenize-batch-size must be greater than zero");
    }

    let tokenizer_load_started_at = Instant::now();
    let tokenizer = args
        .rust_tokenizer_model
        .as_deref()
        .map(|model_dir| RustTextTokenizer::load(model_dir, &args.context_text))
        .transpose()?;
    let tokenizer_load = tokenizer_load_started_at.elapsed();
    if let Some(tokenizer) = tokenizer.as_ref() {
        println!("tokenizer_mode=rust");
        println!(
            "tokenizer_load_ms={:.3}",
            tokenizer_load.as_secs_f64() * 1000.0
        );
        println!("context_token_count={}", tokenizer.context_ids.len());
        println!("text_eos_id={}", tokenizer.text_eos_id);
    } else {
        println!("tokenizer_mode=python_stage0");
    }
    if args.tokenize_only {
        let tokenizer = tokenizer
            .as_ref()
            .context("--tokenize-only requires --rust-tokenizer-model")?;
        let mut texts = if args.text_file.is_some() {
            load_corpus(&args)?
                .into_iter()
                .map(|item| item.text)
                .collect::<Vec<_>>()
        } else {
            vec![args.text.clone()]
        };
        if let Some(batch_size) = args.tokenize_batch_size {
            texts = (0..batch_size)
                .map(|index| texts[index % texts.len()].clone())
                .collect();
        }
        let text_refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        for _ in 0..args.tokenize_warmup_iterations {
            let _ = tokenize_texts(tokenizer, &text_refs, args.tokenize_batch)?;
        }
        let started_at = Instant::now();
        let mut total_target_tokens = 0_usize;
        for _ in 0..args.tokenize_iterations {
            total_target_tokens += tokenize_texts(tokenizer, &text_refs, args.tokenize_batch)?;
        }
        let elapsed = started_at.elapsed();
        let total_texts = text_refs.len() * args.tokenize_iterations;
        println!(
            "tokenize_strategy={}",
            if args.tokenize_batch {
                "batch"
            } else {
                "sequential"
            }
        );
        println!("tokenize_iterations={}", args.tokenize_iterations);
        println!("texts_per_iteration={}", text_refs.len());
        println!("tokenized_texts={total_texts}");
        println!("total_target_tokens={total_target_tokens}");
        println!("tokenization_seconds={:.6}", elapsed.as_secs_f64());
        println!(
            "texts_per_second={:.3}",
            total_texts as f64 / elapsed.as_secs_f64()
        );
        println!(
            "mean_text_us={:.3}",
            elapsed.as_secs_f64() * 1_000_000.0 / total_texts as f64
        );
        if args.text_file.is_none() && args.tokenize_iterations == 1 {
            let request = tokenizer.encode_request(&args.text)?;
            println!(
                "context_token_ids={}",
                serde_json::to_string(request.context_ids)?
            );
            println!(
                "text_token_ids={}",
                serde_json::to_string(&request.text_ids)?
            );
        }
        return Ok(());
    }

    if args.full_pipeline {
        return run_full_pipeline(&args, tokenizer.as_ref()).await;
    }

    let mut engine = connect_engine(&args, &args.handshake_address, "talker").await?;
    let mut tensor_sequence = 0;

    for warmup_index in 0..args.warmup_rounds {
        info!(
            round = warmup_index + 1,
            batch_size = args.batch_size,
            "running warmup wave"
        );
        run_wave(
            &mut engine,
            &args,
            tokenizer.as_ref(),
            warmup_index,
            &mut tensor_sequence,
        )
        .await?;
    }

    let mut waves = Vec::with_capacity(args.rounds);
    for round_index in 0..args.rounds {
        info!(
            round = round_index + 1,
            batch_size = args.batch_size,
            "running measured wave"
        );
        waves.push(
            run_wave(
                &mut engine,
                &args,
                tokenizer.as_ref(),
                args.warmup_rounds + round_index,
                &mut tensor_sequence,
            )
            .await?,
        );
    }

    let wall_seconds = waves
        .iter()
        .map(|wave| wave.wall_time.as_secs_f64())
        .sum::<f64>();
    let requests = waves
        .iter()
        .flat_map(|wave| wave.requests.iter())
        .collect::<Vec<_>>();
    let generated_tokens = requests
        .iter()
        .map(|request| request.generated_tokens)
        .sum::<usize>();
    let measured_tensor_deltas = requests
        .iter()
        .map(|request| request.tensor_deltas)
        .sum::<usize>();
    let audio_frames = requests
        .iter()
        .map(|request| request.generated_tokens.saturating_sub(args.speech_delay))
        .sum::<usize>();
    let audio_seconds =
        audio_frames as f64 * args.frame_stacking_factor as f64 / args.codec_frame_rate;
    let per_request_rtfx = requests
        .iter()
        .map(|request| {
            let frames = request.generated_tokens.saturating_sub(args.speech_delay);
            let audio = frames as f64 * args.frame_stacking_factor as f64 / args.codec_frame_rate;
            audio / request.latency.as_secs_f64()
        })
        .sum::<f64>()
        / requests.len() as f64;

    println!("batch_size={}", args.batch_size);
    println!("rounds={}", args.rounds);
    println!("requests={}", requests.len());
    println!("wall_seconds={wall_seconds:.6}");
    println!(
        "request_throughput_per_second={:.3}",
        requests.len() as f64 / wall_seconds
    );
    println!("generated_tokens={generated_tokens}");
    println!(
        "generated_tokens_per_second={:.3}",
        generated_tokens as f64 / wall_seconds
    );
    println!("multimodal_tensor_deltas={measured_tensor_deltas}");
    println!("estimated_audio_seconds={audio_seconds:.3}");
    println!("aggregate_rtfx={:.3}", audio_seconds / wall_seconds);
    println!("per_request_rtfx_mean={per_request_rtfx:.3}");
    println!(
        "ttft_mean_ms={:.3}",
        requests
            .iter()
            .map(|request| request.ttft.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / requests.len() as f64
    );
    println!(
        "ttft_p95_ms={:.3}",
        percentile_ms(requests.iter().map(|request| request.ttft), 0.95)
    );
    println!(
        "latency_mean_ms={:.3}",
        requests
            .iter()
            .map(|request| request.latency.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / requests.len() as f64
    );
    println!(
        "latency_p95_ms={:.3}",
        percentile_ms(requests.iter().map(|request| request.latency), 0.95)
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    run(Args::parse()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            handshake_address: "tcp://127.0.0.1:62100".to_owned(),
            codec_handshake_address: "tcp://127.0.0.1:62101".to_owned(),
            full_pipeline: false,
            wav_output: PathBuf::from("/tmp/test.wav"),
            sample_rate: 22_050,
            advertised_host: "127.0.0.1".to_owned(),
            text: "hello".to_owned(),
            text_file: None,
            seed: 20_260_729,
            context_text: "[EN]".to_owned(),
            rust_tokenizer_model: None,
            tokenize_only: false,
            tokenize_iterations: 1,
            tokenize_warmup_iterations: 0,
            tokenize_batch: false,
            tokenize_batch_size: None,
            speaker_id: "eng".to_owned(),
            prompt_len: 3,
            max_tokens: 4,
            batch_size: 1,
            warmup_rounds: 0,
            rounds: 1,
            speech_delay: 5,
            frame_stacking_factor: 2,
            codec_frame_rate: 25.0,
            stop_token_id: None,
            audio_temperature: 0.7,
            audio_top_k: 80,
            ready_timeout_secs: 1,
            output_timeout_secs: 1,
            output_dir: None,
        }
    }

    #[test]
    fn omni_request_has_base_fields_plus_additional_information() {
        let request = build_request(&test_args(), "req-1", "hello", None).unwrap();
        let values = value_array(&request).unwrap();
        assert_eq!(values.len(), 21);
        assert_eq!(values[0].as_str(), Some("req-1"));
        assert_eq!(value_array(&values[1]).unwrap().len(), 3);
        assert_eq!(
            map_get(&values[20], "entries")
                .and_then(|entries| map_get(entries, "text"))
                .and_then(|entry| map_get(entry, "scalar_data"))
                .and_then(Value::as_str),
            Some("hello")
        );
    }

    #[test]
    fn request_carries_rust_context_and_target_token_ids() {
        let args = test_args();
        let token_ids = RequestTokenIds {
            context_ids: &[41, 42],
            text_ids: vec![101, 102, 999],
        };
        let request = build_engine_request(
            "req-tokenized",
            args.prompt_len,
            build_sampling_params(&args),
            build_additional_information(
                &args,
                "req-tokenized",
                "hello",
                0,
                Some(token_ids.as_slices()),
            ),
        );
        let values = value_array(&request).unwrap();
        let entries = map_get(&values[20], "entries").unwrap();
        assert_eq!(
            map_get(entries, "context_token_ids")
                .and_then(|entry| map_get(entry, "list_data"))
                .and_then(value_array)
                .unwrap()
                .iter()
                .filter_map(Value::as_u64)
                .collect::<Vec<_>>(),
            [41, 42]
        );
        assert_eq!(
            map_get(entries, "text_tokens")
                .and_then(|entry| map_get(entry, "list_data"))
                .and_then(value_array)
                .unwrap()
                .iter()
                .filter_map(Value::as_u64)
                .collect::<Vec<_>>(),
            [101, 102, 999]
        );
    }

    #[test]
    fn codec_request_targets_stage_one_with_one_placeholder() {
        let request = build_codec_request(&test_args(), "req-codec", "hello");
        let values = value_array(&request).unwrap();
        assert_eq!(value_array(&values[1]).unwrap().len(), 1);
        assert_eq!(
            map_get(&values[20], "entries")
                .and_then(|entries| map_get(entries, "omni_final_stage_id"))
                .and_then(|entry| map_get(entry, "scalar_data"))
                .and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn decodes_float32_audio_tensor() {
        let tensor = OwnedTensor {
            key: "model_outputs".to_owned(),
            dtype: "float32".to_owned(),
            shape: vec![2],
            bytes: [0.25_f32.to_le_bytes(), (-0.5_f32).to_le_bytes()].concat(),
        };
        assert_eq!(tensor_f32_values(&tensor).unwrap(), [0.25, -0.5]);
    }

    #[test]
    fn extracts_omni_multimodal_aux_tensor() {
        let tensor = Value::Array(vec![
            Value::from("int64"),
            Value::Array(vec![Value::from(2), Value::from(3)]),
            Value::from(1),
        ]);
        let frames = vec![Bytes::new(), Bytes::from_static(&[1, 2, 3, 4])];
        let parsed = tensor_from_value("audio".to_owned(), &tensor, &frames).unwrap();
        assert_eq!(parsed.dtype, "int64");
        assert_eq!(parsed.shape, [2, 3]);
        assert_eq!(parsed.bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn corpus_choices_match_cpython_random_choices() {
        let items = (0..10)
            .map(|index| CorpusItem {
                utterance_id: index.to_string(),
                text: index.to_string(),
            })
            .collect::<Vec<_>>();
        let selected = select_corpus_wave(&items, 16, 20_260_729)
            .into_iter()
            .map(|item| item.utterance_id.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(selected, [8, 6, 3, 8, 2, 2, 1, 2, 5, 8, 4, 5, 7, 3, 6, 6]);
    }
}
