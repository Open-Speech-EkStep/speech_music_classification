import torch
from torch.utils.data import DataLoader

from configs.train_config import SpectConfig
from loader.data_loader import SpectrogramDataset, collate_fn
from models.model import Conformer, get_conv_output_sizes

if __name__ == "__main__":
    spect_cfg = SpectConfig()
    songs_dset = SpectrogramDataset('/home/soma/song_speech/speech', 1, spect_cfg)
    print(len(songs_dset))
    feat, label = songs_dset[1000]
    print(feat.shape, label) # [257, T]

    model = Conformer()
    batch = feat.unsqueeze(0)
    print(batch.shape)

    # out = model(batch)
    # print("out shape: ", out.shape)

    lengths = get_conv_output_sizes([feat.shape[1]])
    print('conv out lengths: ', lengths)

    loader = DataLoader(songs_dset, batch_size=10, collate_fn=collate_fn)
    print('data loader len: ', len(loader))

    mini_batch = iter(loader).next()
    out = model(mini_batch[0], mini_batch[1], mini_batch[2])
    print('mini batch output ', out.shape)
    