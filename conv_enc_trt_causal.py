import json
import os
import time

import click
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import requests
import tensorrt as trt
import torch
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from omegaconf.omegaconf import OmegaConf, open_dict


def update_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
            new_state_dict[new_key] = state_dict[key]
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def update_config(model_cfg, codecmodel_path, legacy_codebooks=False):
    ''' helper function to rename older yamls from t5 to magpie '''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print("WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints.")
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2

    return model_cfg


def load_model(nemo_file, hparams_file, codecmodel_path, engine_dir):
    legacy_codebooks = False
    if hparams_file:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        with open_dict(model_cfg):
            model_cfg = update_config(model_cfg, codecmodel_path, legacy_codebooks)

        model = MagpieTTSModel(cfg=model_cfg)
        model.use_kv_cache_for_inference = True

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(nemo_file, weights_only=False)
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        state_dict = update_ckpt(ckpt)
        print(ckpt.keys())
        model.load_state_dict(state_dict)
        checkpoint_name = nemo_file.split("/")[-1].split(".ckpt")[0]
    elif nemo_file is not None:
        model_cfg = MagpieTTSModel.restore_from(nemo_file, return_config=True)
        with open_dict(model_cfg):
            model_cfg = update_config(model_cfg, codecmodel_path, legacy_codebooks)
        model = MagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)
        model.use_kv_cache_for_inference = True
        checkpoint_name = nemo_file.split("/")[-1].split(".nemo")[0]
    else:
        raise ValueError("Need a checkpoint")

    model.cuda()
    model.eval()
    model = model.half()
    return model


class IntEncoder(torch.nn.Module):
    def __init__(self, model, tokenizer_name):
        super().__init__()
        self.tokenizer_name=tokenizer_name
        self.tokenizer=model.tokenizer
        self.bos_id=model.bos_id
        self.eos_id=model.eos_id
        self.text_embedding=model.text_embedding
        self.encoder=model.encoder

        #self.unset_causal_encoding()

        max_length_causal_mask = 2048
        self.encoder_causal_layer_mask = torch.tril(
                torch.ones(max_length_causal_mask, max_length_causal_mask)
                ).view( 1, 1, max_length_causal_mask, max_length_causal_mask).bool().cuda()

    def unset_causal_encoding(self):
        for layer in self.encoder.layers:
            layer.self_attention.is_causal = False
    
    def forward(self,tokens, token_mask, causal_sattn_mask):
        emb_text=self.text_embedding(tokens)
        output=self.encoder(emb_text, token_mask, causal_sattn_mask, None, None, None, None)
        return output

    def export_to_onnx(self, onnx_file, opset_version=17):
        text = "Hello world! How are you doing today?"
        n_batches = 2
        text_encoding = [self.bos_id] + self.tokenizer.encode(text, self.tokenizer_name) + [self.eos_id]
        text_encoding = torch.IntTensor([text_encoding for _ in range(n_batches)]).cuda()
        
        text_lens = torch.IntTensor([text_encoding.shape[1] for _ in range(n_batches)]).cuda()
        max_text_len = torch.max(text_lens).item()
        text_mask = get_mask_from_lengths(text_lens).cuda()  # (B, T)

        padded_text_encoding = stack_tensors(text_encoding,
                                             max_lens=[max_text_len],
                                             pad_value=self.tokenizer.pad)

        causal_sattn_mask = torch.cat([self.encoder_causal_layer_mask[:, :, :max_text_len, :max_text_len] 
                                       for i in range(2)])

        ## Dummy run
        self(text_encoding, text_mask, causal_sattn_mask)

        with torch.no_grad():

            input_names = ["tokens", "token_mask", "causal_sattn_mask"]
            output_names = ["output"]
            dynamic_axes = {
                "tokens": {
                    0: "batch_size",
                    1: "n_texts"
                },
                "token_mask": {
                    0: "batch_size",
                    1: "n_texts"
                },
                "causal_sattn_mask": {
                    0: "batch_size",
                    2: "n_texts",
                    3: "n_texts"
                },
            }
            print(f"{text_encoding.shape=}")
            inputs_args = {
                'tokens': text_encoding,
                'token_mask': text_mask,
                "causal_sattn_mask": causal_sattn_mask

            }
            torch.onnx.export(self,
                              inputs_args,
                              onnx_file,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes,
                              opset_version=17)

            #torch.onnx.export(self, inputs_args, onnx_file, 
            #                  input_names=input_names, dynamic_axes=dynamic_axes,
            #                  output_names=output_names, opset_version=opset_version)

class MagpieEncoderExportTRT:

    def __init__(self,
                 checkpoint_dir,
                 engine_dir,
                 max_seq_len=410,
                 min_seq_len=3,
                 opt_seq_len=None,
                 minBS=1,
                 optBS=None,
                 maxBS=2,
                 dtype="float16"):
        self.checkpoint_dir = checkpoint_dir
        self.engine_dir = engine_dir

        self.encoder_config = {}

        self.dtype = dtype

        if opt_seq_len is None:
            opt_seq_len = min_seq_len + int((max_seq_len - min_seq_len) / 2)

        if optBS is None:
            optBS = minBS + int((maxBS - minBS) / 2)

        if opt_seq_len > max_seq_len or opt_seq_len < min_seq_len:
            raise Exception(
                f"Invalid opt_seq_len should be min_seq_len < opt_seq_len < max_seq_len "
            )

        if optBS > maxBS or optBS < minBS:
            raise Exception(f"Invalid optBS should be minBS < optBS < maxBS")

        self.in_feat_dim = None  # TODO: get from model
        self.num_tokens = None  # TODO: get from model

        self.min_seq_len = min_seq_len
        self.opt_seq_len = opt_seq_len

        self.max_seq_len = max_seq_len

        self.minBS = minBS
        self.optBS = optBS
        self.maxBS = maxBS

        self.encoder_config['min_seq_len'] = min_seq_len
        self.encoder_config['opt_seq_len'] = opt_seq_len
        self.encoder_config['max_seq_len'] = max_seq_len

        self.encoder_config['min_batch_size'] = minBS
        self.encoder_config['opt_batch_size'] = optBS
        self.encoder_config['max_batch_size'] = maxBS
        self.encoder_config['dtype'] = dtype

    def export_encoder_to_onnx(self, model, tokenizer_name="english_phoneme"):
        int_encoder = IntEncoder(model, tokenizer_name)

        onnx_file = os.path.join(self.checkpoint_dir, 'encoder/encoder_grpo.onnx')
       
        self.in_feat_dim = model.text_embedding.embedding_dim  #
        self.num_tokens = len(
            model.tokenizer.tokens) + 2  # add two for bos and eos
        self.encoder_config['num_tokens'] = self.num_tokens
        self.encoder_config['bos_id'] = self.num_tokens - 2
        self.encoder_config['eos_id'] = self.num_tokens - 1
        self.encoder_config['pad_id'] = model.tokenizer.pad

        int_encoder.export_to_onnx(onnx_file)
        enc_graph = gs.import_onnx(onnx.load(onnx_file))
        outputs=enc_graph.outputs
        fix_outputs=[outputs[0]]
        enc_graph.outputs=fix_outputs
        onnx.save(gs.export_onnx(enc_graph), onnx_file)


    def generate_trt_engine(self):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.dtype == "bfloat16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Using FP32")
        
        print(f"{config.flags=}")

        #config.flags = config.flags
        parser = trt.OnnxParser(network, logger)
        onnx_file = os.path.join(self.checkpoint_dir, 'encoder/encoder_grpo.onnx')

        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        nBS = -1
        nFeats = -1
        nMinBS = self.minBS
        nMaxBS = self.maxBS
        nOptBS = self.optBS

        input_feat = network.get_input(0)
        input_mask = network.get_input(1)
        causal_sattn_mask = network.get_input(2)
        input_feat.shape = [nBS, nFeats]
        input_mask.shape = [nBS, nFeats]
        profile.set_shape(
            input_feat.name,
            [nMinBS, self.min_seq_len],
            [nOptBS, self.opt_seq_len],
            [nMaxBS, self.max_seq_len],
        )
        profile.set_shape(
            input_mask.name,
            [nMinBS, self.min_seq_len],
            [nOptBS, self.opt_seq_len],
            [nMaxBS, self.max_seq_len],
        )
        if causal_sattn_mask is not None:
            profile.set_shape(
                causal_sattn_mask.name,
                [nMinBS, 1, self.min_seq_len, self.min_seq_len],
                [nOptBS, 1, self.opt_seq_len, self.opt_seq_len],
                [nMaxBS, 1, self.max_seq_len, self.max_seq_len],
            )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        plan_path = os.path.join(self.engine_dir, "encoder")
        os.makedirs(plan_path, exist_ok=True)

        plan_file = os.path.join(plan_path, 'encoder.plan')
        config_file = os.path.join(plan_path, 'config.json')

        if engineString == None:
            print("Failed building %s" % plan_file)
        else:
            print("Succeeded building %s in %d s" % (plan_file, t1 - t0))
            with open(plan_file, "wb") as f:
                f.write(engineString)
            with open(config_file, 'w') as jf:
                json.dump(self.encoder_config, jf)



@click.command()
@click.option("--dtype", type=str, default="float16", help="dataype of model")
@click.option("--model_ckpt", type=str, help="Path to model checkpoint")
@click.option("--audio_codec", type=str, help="Output Path to audio codec")
@click.option("--hparams_file", type=str, default="", help="Path to hparams file")
@click.option("--max_bs", type=int, default=128, help="maximum batch size")
@click.option("--min_bs", type=int, default=1, help="minimum batch size")
@click.option("--opt_bs", type=int, default=None, help="optimal batch size")
@click.option("--tllm_checkpoint_dir", default="tllm_checkpoint", type=str, help="tllm_ckpt")
@click.option("--engine_dir", default="engine_fp16", type=str, help="engine dir")
def convert_encoder_to_trt(model_ckpt, audio_codec, hparams_file,
                           tllm_checkpoint_dir, engine_dir, dtype,
                           max_bs, min_bs, opt_bs):
    model = load_model(model_ckpt, hparams_file, audio_codec, engine_dir)
    encoder = MagpieEncoderExportTRT(tllm_checkpoint_dir, engine_dir, dtype=dtype,
                                     maxBS=max_bs, minBS=min_bs, optBS=opt_bs)
    encoder.export_encoder_to_onnx(model, tokenizer_name="english_phoneme")
    encoder.generate_trt_engine()


if __name__ == "__main__":
    convert_encoder_to_trt()
