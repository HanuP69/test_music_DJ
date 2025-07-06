import torch
import numpy as np
from miditoolkit import MidiFile
from transformers import T5Tokenizer, T5EncoderModel, GPT2LMHeadModel, EncoderDecoderModel
from miditok import MusicTokenizer
import symusic


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

encoder = T5EncoderModel.from_pretrained("./text2midi_model/encoder").to(device)
decoder = GPT2LMHeadModel.from_pretrained("./text2midi_model/decoder").to(device)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)
model.eval()

text_tokenizer = T5Tokenizer.from_pretrained("./text2midi_model/text_tokenizer")
midi_tokenizer = MusicTokenizer.from_pretrained("./text2midi_model/midi_tokenizer")

bos_token_id = midi_tokenizer["BOS_None"]
eos_token_id = midi_tokenizer["EOS_None"]
pad_token_id = midi_tokenizer["PAD_None"]

model.config.decoder_start_token_id = bos_token_id
model.config.pad_token_id = pad_token_id
model.config.eos_token_id = eos_token_id

def text_to_midi(prompt: str, output_path: str = "output.mid"):

    inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
    

    decoder_input = torch.full(
        (inputs.input_ids.shape[0], 1),
        bos_token_id,
        device=device,
        dtype=torch.long
    )
    

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=decoder_input,
        max_new_tokens=512,
        temperature=0.8,
        top_k=30,
        top_p=0.8,
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        early_stopping=True
    )
    

    tokens = outputs[0].cpu().numpy()
    valid_tokens = [
        t for t in tokens 
        if t < midi_tokenizer.vocab_size 
        and t not in [pad_token_id, bos_token_id]
    ]
    
    if eos_token_id in valid_tokens:
        valid_tokens = valid_tokens[:valid_tokens.index(eos_token_id)]
    
    tokens_2d = np.array(valid_tokens)
    output_symusic = midi_tokenizer.decode(tokens_2d)
    output_symusic.dump_midi(output_path)
    from MIDI_TO_WAV import midi_to_wav
    print(tokens_2d)
    midi_to_wav(midi_path=r"D:\codes\DL\DJ\output.mid", sf2_path=r"D:\codes\DL\DJ\FluidR3_GM\FluidR3_GM.sf2", out_path=r"D:\codes\DL\DJ\op.wav")
   


if __name__ == "__main__":
    text_to_midi(
        "A short classical piece featuring an acoustic guitar, this composition exudes a relaxing and meditative atmosphere. Set in the key of C major with a 2/4 time signature, it unfolds at a moderate tempo. The chord progression of C, G7/B, C, Dm, and G lends a touch of variety to the harmonic landscape, while the overall mood evokes a sense of love and happiness, making it well-suited for a film soundtrack.",
        "output.mid"
    )