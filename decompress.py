import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from utils import init_by_config_path


def decompress(bits, trainer):
    # setup global variables for compressing
    precision = config.ac.precision
    top_value = 2 ** precision
    quarter = top_value // 4
    half = 2 * quarter
    low = 0
    high = top_value
    
    z = 0
    i = 1 # index of bits
    while i <= precision and i <= len(bits):
        if bits[i - 1] == 1:
            z = z + 2**(precision - i)
        i += 1
    # i = precision + 1

    output = [tokenizer.bos_token_id]
    count = 0
    last_token_idx = None
    repeat_count = 0
    
    while (output[-1] != tokenizer.eos_token_id and z != half) or i <= precision + 1:
        # get the probabilities of the next token with ONLY FORWARD PASS
        logits = trainer.predict_step(output[-max_len:])
        logits = logits.detach()
        probs = F.softmax(logits[0][-1], dim=-1)
        probs = probs.to(torch.float64)

        cumprobs = torch.cumsum(probs, dim=0)
        cumprobs = torch.cat((torch.tensor([0.0],device=probs.device), cumprobs), dim=0)
        cumprobs = cumprobs.to(torch.float64)

        # get the target index
        width = high - low + 1
        assert width > 1, 'Precision error when decompressing.'
        widths = width * cumprobs[1:]
        lows = low + widths[:-1].to(torch.int64)
        highs = low + widths[1:].to(torch.int64)

        valid_indices = (lows <= z) & (z < highs)
        if not valid_indices.any():
            print("No valid index found.")
            break
        assert valid_indices.sum() == 1, "More than one valid index found."


        tgt_idx = valid_indices.nonzero(as_tuple=True)[0][0].item() + 1
        decoded_token = tokenizer.decode(tgt_idx)
        # print(tgt_idx, decoded_token)
        if tgt_idx != tokenizer.eos_token_id: 
            decoded_token = decoded_token.replace('\n', '<unk>')
            out_file.write(decoded_token)
        
        ######
        # counter for repeated tokens
        if tgt_idx == last_token_idx:  repeat_count += 1
        else:  repeat_count = 1  # reset the counter if a different token is found
        if repeat_count >= 3:
            print("Detected 3 consecutive tokens, stopping output.")
            break
        last_token_idx = tgt_idx  # update last token index
        ######
        
    
        # backpropagate the target index
        _output = output + [tgt_idx]
        _out_len = min(len(output), max_len)
        _y = _output[-_out_len:]  # pass bos_token
        # _loss = trainer.loss_step(inputs, _y)
        logits = logits.detach()
        # trainer.optim_step(_loss)
        count += 1

        # update the tracker variables
        output = _output
        low = lows[tgt_idx - 1]
        high = highs[tgt_idx - 1]
        
        while high < half or low >= half:
            if high < half:
                low = 2 * low
                high = 2 * high
                z = 2 * z
            elif low >= half:
                low = 2 * (low - half)
                high = 2 * (high - half)
                z = 2 * (z - half)
            if i <= len(bits) and bits[i - 1] == 1:
                z += 1
            i += 1
        while low >= quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            z = 2 * (z - quarter)
            if i <= len(bits) and bits[i - 1] == 1:
                z += 1
            i += 1

    return count


if __name__ == '__main__':
    root_dir_in = 'ECCT_extracted'
    root_dir_out = 'ECCT_restored'
    N, K = 49, 24
    code_name = f'LDPC_K{K}_N{N}'
    channel = 'Rayleigh'  # AWGN or Rayleigh
    filename = 'demo_decode'
    for SNR in tqdm(range(-6,16,3)):
        parser = argparse.ArgumentParser(description='Decompress a binary file while training an LLM.')
        parser.add_argument('--input_file', type=str, \
            default=f'mydata/{root_dir_in}/{code_name}/{channel}/{filename}_SNR_{SNR}.txt')
        parser.add_argument('--output_file', type=str, \
            default=f'mydata/{root_dir_out}/{code_name}/{channel}/{filename}_SNR_{SNR}.txt')
        parser.add_argument('--config_file', type=str, default='config/global/demo.yaml')
        args = parser.parse_args()
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        out_file = open(args.output_file, 'w', encoding='utf-8')

        config, tokenizer, trainer, io = init_by_config_path(
            args.input_file, args.output_file, args.config_file, 'decompress'
        )
        max_len = config.block_size
        print('Start decompressing with the following config...')
        print(config)
        
        with open(args.input_file, 'r') as f:
            bit_streams = f.read().splitlines()  # Split into bitstream lists by line
        
        token_num, bits_num = 0, 0
        for bit_string in bit_streams:
            bits = [int(bit) for bit in bit_string]
            bits_num += len(bits)
            
            count = decompress(bits, trainer)
            out_file.write('\n')
            token_num += count
            
        out_file.close()
        print(f'Bits length: {bits_num}, Decoded {token_num} tokens.')
        print(f'Wrote to file {args.output_file}.')
