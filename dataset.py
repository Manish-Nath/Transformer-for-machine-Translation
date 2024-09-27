import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate if necessary
        enc_input_tokens = enc_input_tokens[:self.seq_len - 2]  # Reserve space for [SOS] and [EOS]
        dec_input_tokens = dec_input_tokens[:self.seq_len - 1]  # Reserve space for [SOS]

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 for [SOS]

        # Create encoder input (with truncation and padding if needed)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Create decoder input (with truncation and padding if needed)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Create label (with truncation and padding if needed)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Ensure the sizes are correct
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,  # seq_len
            'decoder_input': decoder_input,  # seq_len
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# import pandas as pd

# class BilingualDataset(Dataset):
#     def __init__(self, csv_file, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
#         super().__init__()
#         # Load the dataset from the CSV file
#         self.ds = pd.read_csv(csv_file)  # Assuming the CSV file has columns for translations
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#         self.seq_len = seq_len
#         self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
#         self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
#         self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, index):
#         # Access the source and target texts from the DataFrame
#         src_text = self.ds.at[index, self.src_lang]
#         tgt_text = self.ds.at[index, self.tgt_lang]

#         enc_input_tokens = self.tokenizer_src.encode(src_text).ids
#         dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

#         # Truncate if necessary
#         enc_input_tokens = enc_input_tokens[:self.seq_len - 2]  # Reserve space for [SOS] and [EOS]
#         dec_input_tokens = dec_input_tokens[:self.seq_len - 1]  # Reserve space for [SOS]

#         enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 for [SOS] and [EOS]
#         dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 for [SOS]

#         # Create encoder input (with truncation and padding if needed)
#         encoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(enc_input_tokens, dtype=torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         # Create decoder input (with truncation and padding if needed)
#         decoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(dec_input_tokens, dtype=torch.int64),
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         # Create label (with truncation and padding if needed)
#         label = torch.cat(
#             [
#                 torch.tensor(dec_input_tokens, dtype=torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         # Ensure the sizes are correct
#         assert encoder_input.size(0) == self.seq_len
#         assert decoder_input.size(0) == self.seq_len
#         assert label.size(0) == self.seq_len

#         return {
#             'encoder_input': encoder_input,  # seq_len
#             'decoder_input': decoder_input,  # seq_len
#             'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
#             'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
#             'label': label,
#             'src_text': src_text,
#             'tgt_text': tgt_text
#         }

# def causal_mask(size):
#     mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
#     return mask == 0
