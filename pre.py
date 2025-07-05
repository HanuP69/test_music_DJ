
def main():

    import torch
    
    from transformers import AutoTokenizer, T5Tokenizer
    from datasets import load_dataset
    from miditok import REMI, TokenizerConfig
    import mido
    from pathlib import Path

    ds = load_dataset("amaai-lab/MidiCaps", split= 'train').shuffle(seed=42)
    ds = ds.select(range(15000))

    def filterfun(example):
        midi = mido.MidiFile(example['location'])
        print(midi.length)
        if(midi.length>30):
            return False
        else:
            return True
        
        
    


    text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    midi_tokenizer = REMI(TokenizerConfig())

    def fun(batch):
        model_inputs = text_tokenizer(
            batch['caption'],
            max_length = 256,
            padding = "max_length",
            truncation = True
        )

        tokenized_midi = []
        for items in batch['location']:
            tokenized_midi_temp = midi_tokenizer(Path(items))
            tokenized_midi.append(tokenized_midi_temp[0].ids)
        model_inputs["labels"] = tokenized_midi
        return model_inputs
    
    ds = ds.filter(filterfun, num_proc=8)
    ds = ds.map(fun, batch_size=16, batched=True, num_proc=8)
    ds.save_to_disk("dataset")
    text_tokenizer.save_pretrained("text_tokenizer")
    midi_tokenizer.save_pretrained("midi_tokenizer")

if __name__ == "__main__":
    main()

    
