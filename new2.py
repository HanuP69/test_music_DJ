from datasets import load_from_disk
import torch
import os
from transformers import T5Tokenizer, T5EncoderModel
from miditok import MusicTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorWithPadding
from typing import Any, Dict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

ds = load_from_disk("dataset")

text_tokenizer = T5Tokenizer.from_pretrained("text_tokenizer")
midi_tokenizer = MusicTokenizer.from_pretrained("midi_tokenizer/tokenizer.json")

encoder = T5EncoderModel.from_pretrained("google/flan-t5-small")
for param in encoder.parameters():
    param.requires_grad = False

decoder_config = GPT2Config(
    vocab_size=midi_tokenizer.vocab_size,
    n_embd=512,
    n_layer=6,
    n_head=8,
    n_positions=2048,
    is_decoder=True,
    add_cross_attention=True,
    pad_token_id=midi_tokenizer["PAD_None"],
    bos_token_id=midi_tokenizer["BOS_None"],
    eos_token_id=midi_tokenizer["EOS_None"]
)

decoder = GPT2LMHeadModel(decoder_config)

model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.decoder_start_token_id = midi_tokenizer["BOS_None"]
model.config.pad_token_id = midi_tokenizer["PAD_None"]
model.config.eos_token_id = midi_tokenizer["EOS_None"]
model.config.max_length = 1024
model.config.num_beams = 4
model.config.early_stopping = True

class Text2MIDICollator:
    def __init__(self, text_tokenizer, midi_tokenizer, label_pad_token_id=-100, max_length=512):
        self.text_pad = DataCollatorWithPadding(tokenizer=text_tokenizer)
        self.label_pad_token_id = label_pad_token_id
        self.midi_tokenizer = midi_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} 
            for item in batch
        ]
        label_features = [item["labels"] for item in batch]

        for i, item in enumerate(batch):
            input_ids = item["input_ids"]
            labels = item["labels"]
            
            if max(input_ids) >= self.text_tokenizer.vocab_size:
                raise ValueError(f"Batch item {i}: input token {max(input_ids)} >= text vocab size {self.text_tokenizer.vocab_size}")
            if max(labels) >= self.midi_tokenizer.vocab_size:
                raise ValueError(f"Batch item {i}: label token {max(labels)} >= MIDI vocab size {self.midi_tokenizer.vocab_size}")
            if min(input_ids) < 0:
                raise ValueError(f"Batch item {i}: negative input token {min(input_ids)}")
            if min(labels) < 0:
                raise ValueError(f"Batch item {i}: negative label token {min(labels)}")

        input_batch = self.text_pad(input_features)

        max_len = min(max(len(labels) for labels in label_features), self.max_length)
        padded_labels = []
        
        for labels in label_features:

            if len(labels) > self.max_length:
                labels = labels[:self.max_length]

            padded = labels + [self.label_pad_token_id] * (max_len - len(labels))
            padded_labels.append(padded)

        input_batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        

        decoder_input_ids = []
        for labels in label_features:

            if len(labels) > self.max_length:
                labels = labels[:self.max_length]

            decoder_input = [self.midi_tokenizer["BOS_None"]] + labels[:-1]

            if len(decoder_input) > max_len:
                decoder_input = decoder_input[:max_len]

            decoder_input += [self.midi_tokenizer["PAD_None"]] * (max_len - len(decoder_input))
            decoder_input_ids.append(decoder_input)
        
        input_batch["decoder_input_ids"] = torch.tensor(decoder_input_ids, dtype=torch.long)

        return input_batch

data_collator = Text2MIDICollator(
    text_tokenizer=text_tokenizer,
    midi_tokenizer=midi_tokenizer,
    label_pad_token_id=-100,
    max_length=512  )

# Modified training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./text2midi_model",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=512,
    generation_num_beams=2,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=text_tokenizer,  
    train_dataset=ds,
    data_collator=data_collator,
)


trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=text_tokenizer,
        train_dataset=ds,
        data_collator=data_collator,
    )

trainer.train() 
print("Saving model...")
model.save_pretrained("./text2midi_model")

text_tokenizer.save_pretrained("./text2midi_model/text_tokenizer")
midi_tokenizer.save_pretrained("./text2midi_model/midi_tokenizer")

encoder.save_pretrained("./text2midi_model/encoder")
decoder.save_pretrained("./text2midi_model/decoder")

print("Training complete! Model saved in multiple formats.")