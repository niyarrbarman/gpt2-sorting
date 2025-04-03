from transformers import AutoTokenizer
import re
import os
import numpy as np


class Tokenizer:
    def __init__(self):
        self.enc = AutoTokenizer.from_pretrained('./tokenizer/') # Download GPT2 Tokenizer and save it to ./tokenizer/

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)

    @staticmethod
    def read_and_encode_file(file_path, tokenizer):
        all_tokens = []
        lines_encoded = 0
        with open(file_path, "r") as file:
            for line in file:
                tokens = tokenizer.encode(line.strip())
                all_tokens.extend(tokens)
                lines_encoded += 1
                if lines_encoded % 1000 == 0:
                    print(f"{lines_encoded} lines encoded")
        return all_tokens
    
    def tokenize_and_save_shard(self, file_path, output_dir):
        tokens = self.read_and_encode_file(file_path, self)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'sorting_data.npy'), np.array(tokens))
        print(f"loaded {len(np.array(tokens))} tokens")
        print(f"1 epoch = {len(np.array(tokens)) // 524288} steps")
        print(f"Saved tokens shard to {os.path.join(output_dir, 'sorting_data.npy')}")


input_dir = "sorting_fixed_dataset_10m.txt"
output_dir = "./"
tokenizer = Tokenizer().tokenize_and_save_shard(file_path=input_dir, output_dir=output_dir)