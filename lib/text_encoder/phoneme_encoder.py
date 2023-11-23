"""
Directly using https://github.com/xcmyz/FastSpeech/tree/master/text
"""
from typing import List, Optional

from torch import LongTensor

from text import text_to_sequence
from lib.text_encoder.base_encoder import BaseTextEncoder


class PhonemeEncoder(BaseTextEncoder):
    def __init__(self, cleaner_names: Optional[List[str]] = None):
        """
        :param cleaner_names:
            Cleaners are transformations that run over the input text at both training and eval time.

        Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
        hyperparameter. Some cleaners are English-specific. You'll typically want to use:
          1. "english_cleaners" for English text
          2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
             the Unidecode library (https://pypi.python.org/pypi/Unidecode)
          3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
             the symbols in symbols.py to match your data).
        """
        if cleaner_names is None:
            cleaner_names = ['english_cleaners']
        self._cleaner_names = cleaner_names

    def encode(self, text: str) -> LongTensor:
        return text_to_sequence(text, self._cleaner_names)
