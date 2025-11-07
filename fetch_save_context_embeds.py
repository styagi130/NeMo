from nemo.collections.tts.models import MagpieTTSModel
from omegaconf.omegaconf import OmegaConf, open_dict
import os
import glob
import torch
import json
import pathlib
import argparse
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

_REQUIRED_LEN = 109

def get_context_embedding_from_text(text, model):
    context_tokens = model.tokenizer.encode(text, model.text_conditioning_tokenizer_name)#['input_ids']
    print("context tokens", context_tokens)
    print("context tokens", type(context_tokens))
    ## Pad context_tokens
    if len(context_tokens) < _REQUIRED_LEN:
        #_pad_id = model.text_conditioning_tokenizer.pad_token_id
        _pad_id = 0.0
        context_tokens += [_pad_id] * (_REQUIRED_LEN - len(context_tokens))
    context_tokens = context_tokens[:_REQUIRED_LEN]
    context_text_tokens = torch.tensor([context_tokens], dtype=torch.int32, device="cuda")
    context_text_lens = torch.tensor([len(t) for t in context_text_tokens]).cuda()
    context_mask = get_mask_from_lengths(context_text_lens)
    context_input_embedded = model.embed_context_text(context_text_tokens) # (B, L, E)
    context_embeddings = model.context_encoder(context_input_embedded, context_mask, cond=None, cond_mask=None)['output']
    print(context_embeddings.shape, context_text_tokens.shape)
    
    ##Get bos
    audio_codes_bos = torch.full((1, model.num_audio_codebooks, 1),
                                 model.audio_bos_id,
                                 device = context_embeddings.device)
    audio_codes_input = audio_codes_bos
    audio_codes_embedded = model.embed_audio_tokens(audio_codes_input)
    _context_embeddings = torch.cat(
        [context_embeddings, audio_codes_embedded], dim=1
    )

    return _context_embeddings


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
    if hasattr(model_cfg, 'use_local_transformer') and model_cfg.use_local_transformer:
        # For older checkpoints trained with a different parameter name
        model_cfg.local_transformer_type = "autoregressive"
        del model_cfg.use_local_transformer

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
    if hasattr(model_cfg, 'sample_rate'):
        # This was removed from the config and is now in the model class
        sample_rate = model_cfg.sample_rate
        del model_cfg.sample_rate
    else:
        sample_rate = None
    return model_cfg, sample_rate

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

def get_embeddings(
        hparams_file, 
        checkpoint_file, 
        codecmodel_path, 
        text_contexts_file
    ):
    # import ipdb; ipdb.set_trace()
    model_cfg = OmegaConf.load(hparams_file).cfg
    
    if "cfg" in model_cfg:
        model_cfg = model_cfg.cfg
    else:
        model_cfg = model_cfg
    
    with open_dict(model_cfg):
        model_cfg, cfg_sample_rate = update_config(model_cfg, codecmodel_path, legacy_codebooks=False)


    model = MagpieTTSModel(cfg=model_cfg)
    model.use_kv_cache_for_inference = True

    # Load weights from checkpoint file
    print("Loading weights from checkpoint")
    ckpt = torch.load(checkpoint_file, weights_only=False)
    if "state_dict" in ckpt:    
        state_dict = update_ckpt(ckpt["state_dict"])
    model.load_state_dict(state_dict)
    print("Loaded weights.")
    model.cuda()
    model.eval()
    # import ipdb; ipdb.set_trace()
    
    speaker_id_map, context_embeds_map = {}, {}
    delimiter = " | " 

    with text_contexts_file.open() as f:
        context_texts = f.read().strip().split("\n")

    speaker_id_file = text_contexts_file.parent / f"speaker2idmap.json"
    context_embeds_file = text_contexts_file.parent / f"combined_embed.pt"

    languages = set()
    lang2locale_map = {
            "EN": "EN-US",
            "FR": "FR-FR",
            "ES": "ES-US",
            "DE": "DE-DE",
            "ZH": "ZH-CN",
            "VI": "VI-VN",
            "IT": "IT-IT"
            }
    for idx, context_text in enumerate(context_texts):
        context_meta = context_text.split(delimiter)

        lang_speaker = context_meta[2]
        print(f"{context_text}")
        print(f"{lang_speaker}")
        lang = lang_speaker.split()[0].split(":")[1].upper()
        speaker = lang_speaker.split()[2].split(":")[1]
        
        emotion = "" 
        if lang == "ZH":
            emotion = lang_speaker.split()[3]
        else:
            emotion = context_meta[2].split(":")[1]
        emotion = emotion.strip().strip("|").strip()
        speaker = f"{lang2locale_map[lang]}.{speaker}-{emotion}"
        
        languages.add(lang)
        speaker_id_map[idx] = speaker

        context_embed = get_context_embedding_from_text(context_text, model)
        context_embeds_map[idx] = context_embed
        print(f"Finished getting context embeddings for {speaker} with dtype={context_embed.dtype} and shape={context_embed.shape}")
    
    print(f"Saving embeddings to: {str(context_embeds_file)}")
    torch.save(context_embeds_map, context_embeds_file)
    print(f"Saving speaker_name map to: {str(speaker_id_file)}")
    with speaker_id_file.open("w") as f:
        json.dump(speaker_id_map, f)

    print(f"Model bos eos: {model.bos_id} {model.eos_id}")
    print(f"Model audio bos eos: {model.audio_bos_id} {model.audio_eos_id}")
    print(f"Model context bos eos: {model.context_audio_bos_id} {model.context_audio_eos_id}")
    print(f"Languages: {languages}")

    voice_strings = ""
    for id_, name in speaker_id_map.items():
        voice_strings = f"{voice_strings},{name}:{id_}"
    print(voice_strings.strip(","))
    return model

def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_hparams.yaml,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_hparams.yaml")
    parser.add_argument('--checkpoint_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_epoch302.ckpt,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_epoch305.ckpt")
    parser.add_argument('--codecmodel_path', type=str, default="/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo")
    parser.add_argument('--datasets', type=str, default="libri_seen_test,libri_unseen_test")
    parser.add_argument('--base_exp_dir', type=str, default="/datap/misc/eosmount4/AllKernselSize3/NewTransformer")
    parser.add_argument('--draco_exp_dir', type=str, default="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos/experiments/NewT5TTS_FixedPosEmb/AllKernselSize3/NewTransformer")
    parser.add_argument('--server_address', type=str, default="mdesta@login-eos02.eos.clusters.nvidia.com")
    parser.add_argument('--exp_names', type=str, default="multiencoder_small_sp_ks3_lnormapplied")
    parser.add_argument('--local_ckpt_dir', type=str, default="/datap/misc/continuouscheckpoints_fixedposembrough")
    parser.add_argument('--out_dir', type=str, default="/datap/misc/Evals/challenging_fintued_cfg_t_prior_true_multi_noprior_NewTransformerKoelTTS")
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--cfg_scale', type=float, default=2.5)
    parser.add_argument('--apply_attention_prior', action='store_true')
    parser.add_argument('--attention_prior_epsilon', type=float, default=0.1)
    parser.add_argument('--attention_prior_lookahead_window', type=int, default=5)
    parser.add_argument('--estimate_alignment_from_layers', type=str, default="4,6")
    parser.add_argument('--apply_prior_to_layers', type=str, default="2,4,6,8,10")
    parser.add_argument('--start_prior_after_n_audio_steps', type=int, default=0)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--text_contexts_file', type=pathlib.Path, required=True)
    args = parser.parse_args()

    estimate_alignment_from_layers = None
    if args.estimate_alignment_from_layers is not None:
        estimate_alignment_from_layers = [int(l.strip()) for l in args.estimate_alignment_from_layers.split(",")]
    apply_prior_to_layers = None
    if args.apply_prior_to_layers is not None:
        apply_prior_to_layers = [int(l.strip()) for l in args.apply_prior_to_layers.split(",")]

    if (args.hparams_files is not None) and (args.checkpoint_files is not None) and (args.hparams_files != "null"):
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")
        print("Running inference for hparams files: ", hparam_files)
        print("Running inference for checkpoint files: ", checkpoint_files)
        assert len(hparam_files) == len(checkpoint_files), "Number of hparams files and checkpoint files should be the same."
        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            model = get_embeddings(
                hparams_file=hparams_file, 
                checkpoint_file=checkpoint_file,
                codecmodel_path=args.codecmodel_path,
                text_contexts_file=args.text_contexts_file
            )
        return model

if __name__ == '__main__':
    model = main()

