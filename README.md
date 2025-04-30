# LLM-SSCC

Codes for "Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking"
=======
## Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking

Tianqi Ren, Rongpeng Li, Ming-min Zhao, Xianfu Chen, Guangyi Liu, Yang Yang, Zhifeng Zhao and Honggang Zhang

This is the implementation of the contributed paper: 

Ren T, Li R, Zhao M, et al. Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking[J]. arXiv preprint arXiv:2501.04285, 2025.

The modules are:

1. `compress.py`: The script to compress a text file by the arithmetic encoding algorithm with a pretrained large language model (LLM), e.g. GPT-2.
2. `dataloader.py`: The DataLoader class for loading the text file.
3. `decompress.py`: The script to decompress a binary file by the arithmetic decoding algorithm with a pretrained LLM identical to the compression model.
4. `ECCT_forward.py`: The script to evaluate a trained ECCT model on encoded messages transmitted through different channels, calculating and logging metrics such as BER and FER for various SNR levels.
5. `extract_message.py`: The script to extract message from codeword decoded by ECCT.
6. `file_io.py`: The FileIO class for reading and writing bits to either text or binary files.
7. `Model.py`: The ECCT model class.
8. `trainer.py`: The Trainer class for updating the model parameters and predicting next-token with a pretrained LLM.
9. `utils.py`: Utility funtions.

---

### Requirements

- See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.

---

### Usage

#### [Compress]

```python
python compress.py
```

#### [Transmit through channel and ECCT]

```python
python ECCT_forward.py
```

#### [Decompress]

```python
python decompress.py
```

---

### Bibtex

@misc{ren2025separatesourcechannelcoding,
      title={Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking}, 
      author={Tianqi Ren and Rongpeng Li and Ming-min Zhao and Xianfu Chen and Guangyi Liu and Yang Yang and Zhifeng Zhao and Honggang Zhang},
      year={2025},
      eprint={2501.04285},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2501.04285}, 
}



