from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx: int):
        """
        {
            'mel': Tensor of shape (freq, T),
            'text': str,
            'alignment': LongTensor of shape (P,),
            'pitch': Tensor of shape (T,),
            'energy': Tensor of shape (T,),
            'wave': Tensor of shape (1, wave_len),
            'phonemes_tokens': LongTensor of shape (P,),
        }
        """
        raise NotImplementedError
