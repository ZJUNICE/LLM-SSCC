# Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking

This repository, corresponding to the paper https://arxiv.org/abs/2501.04285, demonstrates the use of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, which together constitute an LLM-based Separate Source Channel Coding (LLM-SSCC) system, offering superior performance over Joint Source Channel Coding (JSCC) schemes.

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

## Paper

- T. Q. Ren, R. P. Li, M. M. Zhao, et al., “Separate source channel coding is still what you need: an LLM-based rethinking,” ZTE Communications, vol. 23, no. 1, pp. 30–44, Mar. 2025. doi: 10.12142/ZTECOM.202501005.

## Files Overview

1. `compress.py`: The script to compress a text file by the arithmetic encoding algorithm with a pretrained large language model (LLM), e.g. GPT-2.
2. `dataloader.py`: The DataLoader class for loading the text file.
3. `decompress.py`: The script to decompress a binary file by the arithmetic decoding algorithm with a pretrained LLM identical to the compression model.
4. `ECCT_forward.py`: The script to evaluate a trained ECCT model on encoded messages transmitted through different channels, calculating and logging metrics such as BER and FER for various SNR levels.
5. `extract_message.py`: The script to extract message from codeword decoded by ECCT.
6. `file_io.py`: The FileIO class for reading and writing bits to either text or binary files.
7. `Model.py`: The ECCT model class.
8. `trainer.py`: The Trainer class for updating the model parameters and predicting next-token with a pretrained LLM.
9. `utils.py`: Utility funtions.

## Setup Instructions

### Requirements

- See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.

---

### Usage

1. **Compress**: To implement source encoding for text data using arithmetic coding combined with LLM for probability estimation.

   ```'python
   python compress.py
   ```

2. **Transmit through channel and ECCT**: To implement a process where messages undergo channel encoding and are transmitted through channel. The transmitted signals are then processed by ECCT to predict the original codewords.

   ```python
   python ECCT_forward.py
   ```

3. **Extract message from codeword**: To extract message from codeword predicted by ECCT, obtaining the recovered information.

   ```python
   python extract_message.py
   ```

4. **Decompress**: To implement source decoding for binary data using arithmetic coding combined with LLM for probability estimation.

   ```python
   python decompress.py
   ```

