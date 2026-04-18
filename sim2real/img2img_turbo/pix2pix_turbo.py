# Vendored from: https://github.com/GaParmar/img2img-turbo
# Upstream commit: 86f54146590ffb4543c8cf85b5a36657da670924
# Original path in upstream: src/pix2pix_turbo.py
#
# Minimal edits for this repo:
# - Replaced `sys.path` hack with package-relative import of scheduler/VAE helpers.
# - Added `device=` so weights are not forced onto CUDA only (inference may use CPU).
# - `forward()` moves token tensors to the same device as `text_encoder`.
# - After loading a `.pkl`, copy LoRA metadata into `self` when present so `save_model()` matches upstream.

import os
import copy
import requests
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig

from .sd_turbo_helpers import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


def _assign_lora_meta_from_sd(self, sd: dict) -> None:
    if not isinstance(sd, dict):
        return
    if "unet_lora_target_modules" in sd:
        self.target_modules_unet = sd["unet_lora_target_modules"]
    if "vae_lora_target_modules" in sd:
        self.target_modules_vae = sd["vae_lora_target_modules"]
    if "rank_unet" in sd:
        self.lora_rank_unet = sd["rank_unet"]
    if "rank_vae" in sd:
        self.lora_rank_vae = sd["rank_vae"]


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super().__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * self.r


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
        device=None,
    ):
        super().__init__()
        if device is None:
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self._torch_device = device
        else:
            self._torch_device = torch.device(device)
        dev = self._torch_device

        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).to(dev)
        self.sched = make_1step_sched(dev)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(dev)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(dev)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(dev)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(dev)
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            _assign_lora_meta_from_sd(self, sd)

        elif pretrained_name == "sketch_to_image_stochastic":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            _assign_lora_meta_from_sd(self, sd)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"]
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            _assign_lora_meta_from_sd(self, sd)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = [
                "conv1",
                "conv2",
                "conv_in",
                "conv_shortcut",
                "conv",
                "conv_out",
                "skip_conv_1",
                "skip_conv_2",
                "skip_conv_3",
                "skip_conv_4",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ]
            vae_lora_config = LoraConfig(
                r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "conv",
                "conv1",
                "conv2",
                "conv_shortcut",
                "conv_out",
                "proj_in",
                "proj_out",
                "ff.net.2",
                "ff.net.0.proj",
            ]
            unet_lora_config = LoraConfig(
                r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        unet.to(dev)
        vae.to(dev)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=dev).long()
        self.text_encoder.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._torch_device

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        te_dev = self.text_encoder.device
        if prompt is not None:
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(te_dev)
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens.to(te_dev))[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
