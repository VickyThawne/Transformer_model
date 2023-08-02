import os

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tiktoken

from generation_model import BigramLanguageModel, Configuration


dataset = 'sherlock'

model_path = "models/model.pth"

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 4 # what is the maximum context length for predictions?
eval_iters = 40


learning_rate = 1e-4 # max learning rate
max_iters = 400 # total number of training iterations
weight_decay = 1e-1

device = "cuda" if torch.cuda.is_available() else "cpu"


best_val_loss = float('inf')
wait = 0

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":

    config =  Configuration()
    model = BigramLanguageModel(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("File not found, starting from scratch.")



    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(train_data, dtype=np.uint16, mode='r')
    # val_data = np.memmap(val_data, dtype=np.uint16, mode='r')
    print("data Loaded successfully...")
    # train_data, val_data = load_data(Path('data/sherlocks_diary.txt'))

    print("initializing training...")
    for iter_num in tqdm(range(max_iters)):

        if iter_num % eval_iters == 0 or iter_num == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        loss += 0.01 * torch.sum(torch.pow(model.lm_head.weight, 2))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Early stopping
        if loss < best_val_loss:
            best_val_loss = loss
            wait = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping at iteration {}.".format(iter_num))
                break

    torch.save(model.state_dict(), 'article_generation_model.pth')

