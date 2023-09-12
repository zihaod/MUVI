import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from muvi.common.registry import registry
from muvi.models.llama_model import LlamaForCausalLM
from muvi.models.base_model import BaseModel
from transformers import LlamaTokenizer, Wav2Vec2FeatureExtractor, AutoModel

import torchaudio.transforms as T


@registry.register_model("muvi")
class MUVI(BaseModel):
    """
    MERT GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/muvi.yaml",
    }

    def __init__(
        self,
        mert_model="m-a-p/MERT-v1-330M",
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.low_resource = low_resource

        print('Loading Audio Encoder')
        self.audio_encoder = AutoModel.from_pretrained(mert_model, trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model, trust_remote_code=True)

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder = self.audio_encoder.eval()

        print('Loading Audio Encoder Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.audio_encoder.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<AudioHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def audioenc_to_cpu(self):
        self.audio_encoder.to("cpu")
        self.audio_encoder.float()

    def encode_audio(self, audio, attn=None):
        device = audio.device
        if self.low_resource:
            self.audioenc_to_cpu()
            audio = audio.to("cpu")

        #with self.maybe_autocast():
        if attn is None:
            #audio_embeds = self.audio_encoder(input_values=audio)['last_hidden_state']#.to(device)
            audio_embeds = torch.stack(self.audio_encoder(input_values=audio, 
                                                          output_hidden_states=True).hidden_states) # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3) #[B, T, 1024]

        else:
            #audio_embeds = self.audio_encoder(input_values=audio, attention_mask=attn)['last_hidden_state']
            audio_embeds = torch.stack(self.audio_encoder(input_values=audio, 
                                                          output_hidden_states=True, 
                                                          attention_mask=attn).hidden_states) # [25, B, T, 1024]
            audio_embeds = audio_embeds.transpose(0, 1).mean(-3) #[B, T, 1024]
            
        #Average time steps:
        t = 325
        B, T, D = audio_embeds.shape
        avg_tmp = audio_embeds[:, :T//t*t].reshape(B, T//t, t, D).mean(2)
        #Average the remaining steps
        if T % t > 0:
          avg_last = audio_embeds[:, T//t*t:].reshape(B, 1, T%t, D).mean(2)
          audio_embeds = torch.concat([avg_tmp, avg_last], dim=1)
        else:
          audio_embeds = avg_tmp
        audio_embeds = audio_embeds.to(device)
        inputs_llama = self.llama_proj(audio_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(audio.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before, p_after = prompt.split('<AudioHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = atts_audio[:, :1].expand(-1, wrapped_audio_embeds.shape[1])
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio
        
    def instruction_prompt_wrap(self, audio_embeds, atts_audio, prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_before = []
            p_after = []

            for i in range(batch_size):
                p_b, p_a = prompt[i].split('<AudioHere>')
                p_before.append(p_b)
                p_after.append(p_a)
  
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", padding='longest', add_special_tokens=False).to(audio_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", padding='longest', add_special_tokens=False).to(audio_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            wrapped_audio_embeds = torch.cat([p_before_embeds, audio_embeds, p_after_embeds], dim=1)
            wrapped_atts_audio = torch.cat([p_before_tokens.attention_mask, atts_audio, p_after_tokens.attention_mask], dim=1)
            return wrapped_audio_embeds, wrapped_atts_audio
        else:
            return audio_embeds, atts_audio

    def generation_instruction_prompt_wrap(self, audio_embeds, atts_audio prompt):
        if prompt:
            batch_size = audio_embeds.shape[0]
            p_befores = []
            p_afters = []
    
            wrapped_audio_embeds = []
            wrapped_atts_audios = []
            
            for i in range(batch_size):
                p_b, p_a = prompt[i].split('<AudioHere>')
                p_befores.append(p_b)
                p_afters.append(p_a)
            
            for i in range(batch_size):
                p_before = p_befores[i]
                p_after = p_afters[i]
                p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
                p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(audio_embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).squeeze(0)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).squeeze(0)
                #print(p_before_embeds.shape, p_after_embeds.shape, audio_embeds[i].shape)
                bos = torch.ones([1, 1], dtype=torch.int64, device=audio_embeds.device) * self.llama_tokenizer.bos_token_id
                bos_embed = self.llama_model.model.embed_tokens(bos).squeeze(0)
                atts_bos = torch.ones(1).to(torch.int64).to(audio_embeds.device)
                
                wrapped_audio_embed = torch.cat([bos_embed,p_before_embeds, audio_embeds[i], p_after_embeds], dim=0) # 2D tensor
                wrapped_atts_audio = torch.cat([atts_bos,
                                                p_before_tokens.attention_mask.squeeze(0), 
                                                atts_audio[i], 
                                                p_after_tokens.attention_mask.squeeze(0)], dim=0)

                wrapped_atts_audios.append(wrapped_atts_audio)
                wrapped_audio_embeds.append(wrapped_audio_embed)
         
           
            max_len = max([embed.shape[0] for embed in wrapped_audio_embeds])
            padded_audio_embeds = []
            padded_atts_audios = []
    
            for i in range(len(wrapped_audio_embeds)):
                embed = wrapped_audio_embeds[i]
                atts = wrapped_atts_audios[i]
                
                pad_len = max_len - embed.shape[0]
                pad_embed = self.llama_model.model.embed_tokens(2 * torch.ones(pad_len).to(torch.int64).to(audio_embeds.device))
                pad_atts = torch.zeros(pad_len).to(torch.int64).to(audio_embeds.device)
    
                padded_audio_embeds.append(torch.cat([pad_embed, embed], dim=0))
                padded_atts_audios.append(torch.cat([pad_atts, atts], dim=0))
    
            padded_audio_embeds = torch.stack(padded_audio_embeds)
            padded_atts_audios = torch.stack(padded_atts_audios)
            #print(padded_audio_embeds.shape)
    
            return padded_audio_embeds, padded_atts_audios
        else:
            return audio_embeds, atts_audio

    def forward(self, samples):
        audio = samples["audio"]
        attn = samples["attention_mask"] if "attention_mask" in samples else None
        audio_embeds, atts_audio = self.encode_audio(audio, attn)
        if 'instruction_input' in samples:  # instruction dataset
            #print('Instruction Batch')
            instruction_prompt = []
            for instruction in samples['instruction_input']:
                prompt = '<Audio><AudioHere></Audio> ' + instruction
                instruction_prompt.append(self.prompt_template.format(prompt))
            audio_embeds, atts_audio = self.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            audio_embeds, atts_audio = self.prompt_wrap(audio_embeds, atts_audio, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(audio.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_audio.shape[0], atts_audio.shape[1]+1],
                       dtype=torch.long).to(audio.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = audio_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_audio[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, audio_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_audio, to_regress_tokens.attention_mask], dim=1)

        #with self.maybe_autocast():
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        mert_model = cfg.get("mert_model", "")
        llama_model = cfg.get("llama_model")

        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            mert_model=mert_model,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MUVI
        if ckpt_path:
            print("Load MERT-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
