import torch
from model import GPT, GPTConfig
import torch.nn.functional as F
from tokenize_data import Tokenizer
from infer import generate
import numpy as np
import time

# -----------------------------------------------------------------------------

class DataLoader:
    def __init__(self, B, T, path):
        self.B = B
        self.T = T
        tokens = np.load(path)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

total_batch_size = 524288 
B = 16 
T = 1024
assert total_batch_size % (B * T) == 0, "total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total batch size: {total_batch_size}")
print(f"=> gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
print(model)
print("Model loaded :D")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

model.to(device)

# -----------------------------------------------------------------------------

######## Hyperparameters ########

warmup_steps = 500
max_steps = 10_000
max_lr = 6e-4
min_lr = max_lr * 0.1

train_loader = DataLoader(B=B, T=T, path='sorting_data.npy')

import math

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# -----------------------------------------------------------------------------

####### Checkpoint #########

import os

def check_checkpoints(folder_path):
    load_checkpoint = False
    checkpoint_path = None

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        checkpoint_files = [f for f in os.listdir(folder_path) if f.startswith("checkpoint_") and f.endswith(".pth")]

        if checkpoint_files:
            load_checkpoint = True
            largest_checkpoint = max(
                checkpoint_files,
                key=lambda x: int(x.split("_")[1].split(".")[0])
            )
            checkpoint_path = os.path.join(folder_path, largest_checkpoint)

    return load_checkpoint, checkpoint_path

folder = "checkpoints"
load_checkpoint, checkpoint_path = check_checkpoints(folder)
last_step = 0

if load_checkpoint:
    checkpoint = torch.load(checkpoint_path)
    state_dict_model = checkpoint["model"]
    state_dict_optim = checkpoint["optimizer"]
    last_step = checkpoint['step']
    print(f"Loading Model from checkpoint :: {last_step-1}")

    model.load_state_dict(state_dict_model)
    print("Succesfully loaded Model")
    optimizer.load_state_dict(state_dict_optim)
    print("Succesfully loaded Optimizer")

print(f"Starting from step {last_step}")

# -----------------------------------------------------------------------------

####### Training Loop #########

loss_plt = []

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_processed = (train_loader.B * train_loader.T * grad_accum_steps) / (t1-t0)

    print(f"step {last_step+step:4d} | loss: {loss_accum.item()} |lr: {lr:.4e} | norm: {norm:.4f} | dt: {(t1-t0)*1000}ms | tokens/sec: {tokens_processed:.2f}")
    loss_plt.append(loss_accum.item())

    if (step+1) % 200 == 0:
        prompt = "[[-203.16, -694.2, -749.83, 218.17, 492.02, 119.5, -851.63, 662.96, 858.98, 380.3], "
        generated_text = generate(model, prompt, max_length=150, num_return_sequences=1, device=device)
        print(f'\n\n> test_output_{step+1}: {generated_text[0]}\n\n')

    if (step+1) % 5_000 == 0:
        checkpoint = {'step': last_step + step + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }
        torch.save(checkpoint, f'checkpoints/checkpoint_{last_step+step+1}.pth')
        print(f"Saved checkpoint {step+1}")

# -----------------------------------------------------------------------------

os.makedirs('plots', exist_ok=True)
np.save(f'plots/loss_{last_step+step+1}.npy', np.array(loss_plt))