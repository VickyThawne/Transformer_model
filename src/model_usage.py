import tiktoken
import torch

def main():
    enc = tiktoken.get_encoding("gpt2")
    model = BigramLanguageModel()
    model = model.to(device)

    start = "today is the perfect day for a"
    target_length = 100

    total_generated_tokens = enc.encode(start, allowed_special={""})

    with torch.no_grad():
        with ctx:
            for k in range(target_length % 16):
                start_ids = total_generated_tokens[-16:]
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                y = model.generate(x, max_new_tokens=16, temperature=0.5)
                new_x = y[0].tolist()

                total_generated_tokens += new_x

    total_generated = enc.decode(total_generated_tokens)
    print(total_generated)

if __name__ == "__main__":
    main()
