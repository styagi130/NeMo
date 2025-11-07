from numpy import who
import nemo
import torch
import argparse
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
import soundfile as sf
import numpy as np
from typing import Tuple
import onnx
import onnxruntime as ort


class AudioCodecModel_ONNX_SUP(AudioCodecModel):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer)
    
    def decode_audio(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_len = tokens_len.squeeze(1)
        dequantized = self.dequantize(tokens=tokens, tokens_len=tokens_len)
        dequantized = dequantized.to(self.dtype) # make sure that the dequantized is in the model dtype
        # Apply decoder to obtain time-domain audio for each frame
        audio, audio_len = self.audio_decoder(inputs=dequantized, input_len=tokens_len)
        audio_len = audio_len.unsqueeze(1)
        return audio, audio_len

    def encode_audio_(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_len = audio_len.squeeze(1)
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)

        tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
        encoded_len = encoded_len.unsqueeze(1).to(torch.int32)
        return tokens, encoded_len


def load_audio(audio_filepath, sample_rate=22050, pad_multiple=1024):
    features = AudioSegment.segment_from_file(
        audio_filepath, target_sr=sample_rate, n_segments=-1, trim=False,
    )
    audio_samples = features.samples
    audio = torch.tensor(audio_samples)
    audio = torch.nn.functional.pad(audio, (0, pad_multiple - audio.size(0) % pad_multiple), value=0).unsqueeze(0).cuda()
    audio_length = torch.tensor([audio.size(1)]).long().cuda()
    return audio, audio_length

def get_codes(model, audio, audio_lengths):
    codes, codes_length = model.encode(audio=audio, audio_len=audio_lengths)
    return codes, codes_length

def export_encoder(model, audio, audio_length, encoder_path):
    inputs = {
            "audio": audio,
            "audio_len": audio_length.unsqueeze(1)
            }
    input_args = tuple(inputs.values())
    input_names = list(inputs.keys())
    output_names = ["codes", "codes_len"]
    dynamic_axes = {
            "audio":{
                0: "batch_size",
                1: "n_time"
                },
            "audio_len":{
                0: "batch_size"
                }
            }
    export_options = torch.onnx.ExportOptions(
        dynamic_shapes=True,
        diagnostic_options=torch.onnx.DiagnosticOptions(verbosity_level=30),
    )
    #encoder_exported = torch.onnx.dynamo_export(
    #        model,
    #        model_kwargs=inputs,
    #        export_options=export_options
    #        )
    #encoder_exported.save(encoder_path)
    cfg = model.audio_encoder
    print(type(cfg))
    cfg = model.audio_decoder
    print(type(cfg))
    #print(cfg)
    model.forward = model.encode_audio_
    torch.onnx.export(
            model, #.encode,
            args=input_args,
            f=encoder_path,
            opset_version=17,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
            )
    print("Exported encoder, saved to:", encoder_path)
    

def export_decoder(model, codes, codes_length, decoder_path):
    inputs = (
            codes,
            codes_length.unsqueeze(1).to(torch.int32)
            )
    input_names = [
            "codes", "codes_length"
            ]
    output_names = ["audio", "audios_len"]
    dynamic_axes = {
            "codes":{
                0: "batch_size",
                2: "n_time"
                },
            "codes_length":{
                0: "batch_size"
                }
            }

    model.forward = model.decode_audio
    torch.onnx.export(
            model,
            args=inputs,
            f=decoder_path,
            opset_version=17,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
            )
    print("Exported decoder, saved to:", decoder_path)

def main(codec_model_path, audio_filepath, out_encoder_path, out_decoder_path):
    ## Load codec model path
    model = AudioCodecModel_ONNX_SUP.restore_from(codec_model_path, strict=False).cuda().eval()
    print(type(model), codec_model_path)
    with torch.no_grad():
        audio, audio_length = load_audio(audio_filepath)
        codes, codes_length = get_codes(model, audio, audio_length)
        audio_length = audio_length
        codes_length = codes_length

        ## Export Encoder
        print("============Exporting Encoder===============")
        print(type(model.audio_encoder))
        print(type(model.audio_decoder))
        export_encoder(model, audio, audio_length, out_encoder_path)
        print("============Exporting decoder===============")
        model = model.half()
        export_decoder(model, codes, codes_length, out_decoder_path)
        
        ## Test encoder
        print("================Verifying encoder=================")
        enc_model = onnx.load(out_encoder_path)
        onnx.checker.check_model(enc_model)
        options = ort.SessionOptions()
        enc_sess = ort.InferenceSession(out_encoder_path, sess_options=options, providers=["CUDAExecutionProvider"])
        print("Encoder session inputs", out_encoder_path)
        for out in enc_sess.get_inputs():
            print(out.name, out.shape, out.type)

        print("Encoder session outputs", out_encoder_path)
        for out in enc_sess.get_outputs():
            print(out.name, out.shape, out.type)
        
        ## Verify encoder
        enc_ips = {
                "audio": audio.detach().cpu().numpy(),
                "audio_len": audio_length.unsqueeze(1).detach().cpu().numpy()
                }
        enc_outputs = enc_sess.run(None, enc_ips)
        codes = enc_outputs[0]
        codes_length = enc_outputs[1].reshape(-1, 1)
        sf.write("codes_onnx/original.wav", audio.squeeze(0).detach().cpu().numpy(), 22050)

        ## Test decoder
        print("================Verifying decoder=================")
        dec_model = onnx.load(out_decoder_path)
        onnx.checker.check_model(dec_model)

        options = ort.SessionOptions()
        dec_sess = ort.InferenceSession(out_decoder_path, sess_options=options, providers=["CUDAExecutionProvider"])
        print("Decoder session inputs", out_decoder_path)
        for out in dec_sess.get_inputs():
            print(out.name, out.shape, out.type)

        print("Decoder session outputs", out_decoder_path)
        for out in dec_sess.get_outputs():
            print(out.name, out.shape, out.type)

        dec_inputs = {
                "codes": codes,
                "codes_length": codes_length
                }
        dec_output = dec_sess.run(None, dec_inputs)
        audio = dec_output[0]
        audio = audio.astype(np.float32)
        sf.write("codes_onnx/decoded.wav", audio.reshape((-1)), 22050)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""
    Use this script to generate audio codec onnx checkpoint.""")
    parser.add_argument("--codec_model_path", required=True, help="Codec model path.")
    parser.add_argument("--audio_filepath", required=True, help="Audio filepath.")
    parser.add_argument("--out_encoder", required=True, help="Encoder write path.")
    parser.add_argument("--out_decoder", required=True, help="Decoder write path.")
    args = parser.parse_args()

    main(args.codec_model_path, 
         args.audio_filepath,
         args.out_encoder,
         args.out_decoder)
