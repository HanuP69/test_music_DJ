import torch
from transformers import T5Tokenizer, EncoderDecoderModel
from miditok import MusicTokenizer
from pathlib import Path

model = EncoderDecoderModel.from_pretrained("./text2midi_model").to("cuda")
text_tokenizer = T5Tokenizer.from_pretrained("./text2midi_model/text_tokenizer")
midi_tokenizer = MusicTokenizer.from_pretrained("./text2midi_model/midi_tokenizer")

bos_token_id = midi_tokenizer["BOS_None"]
eos_token_id = midi_tokenizer["EOS_None"]
pad_token_id = midi_tokenizer["PAD_None"]

model.config.decoder_start_token_id = bos_token_id
model.config.pad_token_id = pad_token_id
model.config.eos_token_id = eos_token_id

def text_to_midi(prompt: str, output_path: str = "output.mid"):
    
    inputs = text_tokenizer(prompt, return_tensors="pt").to("cuda")
    
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        early_stopping=True,
        decoder_input_ids = midi_tokenizer["BOS_None"]
    )
    
    generated_tokens = outputs[0].cpu().numpy()
    valid_tokens = [t for t in generated_tokens if t < midi_tokenizer.vocab_size]
    
    if eos_token_id in valid_tokens:
        valid_tokens = valid_tokens[:valid_tokens.index(eos_token_id)]
    
    if valid_tokens:
        midi = midi_tokenizer.decode(valid_tokens)
        midi.dump(output_path)
        print(f"✅ Generated MIDI saved to {output_path}")
        return True
    else:
        print("❌ No valid tokens generated")
        return False

text_to_midi(
    "a fast-paced electronic drum solo with heavy bass",
    "drum_solo.mid"
)