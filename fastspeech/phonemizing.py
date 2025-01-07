
from phonemizer import phonemize
from phonemizer.separator import Separator



# Example IPA phoneme vocabulary
# symbols = [
#     "p", "b", "t", "d", "ʈ", "ɖ", "c", "ɟ", "k", "g", "q", "ɢ", "ʔ",
#     "m", "ɱ", "n", "ɳ", "ɲ", "ŋ", "ɴ",
#     "ʙ", "r", "ʀ",
#     "ⱱ", "ɾ", "ɽ",
#     "ɸ", "β", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ç", "ʝ", "x", "ɣ", "χ", "ʁ", "ħ", "ʕ", "h", "ɦ",
#     "ɬ", "ɮ",
#     "ʋ", "ɹ", "ɻ", "j", "ɰ",
#     "l", "ɭ", "ʎ", "ʟ",
#     "ɓ", "pʼ", "ǀ", "ɗ", "tʼ", "ǃ", "ʄ", "kʼ", "ǂ", "ɠ", "sʼ", "ǁ", "ʛ", "ʼ",
#     "ʍ", "w", "ɥ", "ʜ", "ʢ", "ʡ", "ɕ", "ʑ", "ɺ", "ɧ",
#     "t͡s", "t͡ʃ", "t͡ɕ", "ʈ͡ʂ", "d͡z", "d͡ʒ", "d͡ʑ", "ɖ͡ʐ"
# ]

# symbol_dict = {
#     # Vowels
#     "i": 1, "y": 2, "ɪ": 3, "ʏ": 4, "e": 5, "ø": 6, "ɛ": 7, "œ": 8,
#     "æ": 9, "a": 10, "ɑ": 11, "ɒ": 12, "ʌ": 13, "ɔ": 14, "o": 15,
#     "u": 16, "ʊ": 17, "ɯ": 18, "ɤ": 19, "ə": 20, "ɜ": 21, "ɝ": 22,
#     "ɞ": 23, "ɚ": 24, "ʉ": 25, "ɨ": 26, "ɐ": 27, "ʊ̈": 28,

#     # Plosives
#     "p": 29, "b": 30, "t": 31, "d": 32, "k": 33, "g": 34, "ʔ": 35,
#     "q": 36, "ɢ": 37, "ʡ": 38,

#     # Nasals
#     "m": 39, "ɱ": 40, "n": 41, "ɳ": 42, "ŋ": 43, "ɴ": 44,

#     # Trills
#     "ʙ": 45, "r": 46, "ʀ": 47,

#     # Taps and Flaps
#     "ɾ": 48, "ɽ": 49,

#     # Fricatives
#     "f": 50, "v": 51, "θ": 52, "ð": 53, "s": 54, "z": 55, "ʃ": 56, "ʒ": 57,
#     "ʂ": 58, "ʐ": 59, "ç": 60, "ʝ": 61, "x": 62, "ɣ": 63, "χ": 64, "ʁ": 65,
#     "ħ": 66, "ʕ": 67, "h": 68, "ɦ": 69,

#     # Approximants
#     "ʋ": 70, "ɹ": 71, "ɻ": 72, "j": 73, "ɰ": 74,

#     # Lateral Approximants
#     "l": 75, "ɭ": 76, "ʎ": 77, "ʟ": 78,

#     # Clicks
#     "ʘ": 79, "ǀ": 80, "ǃ": 81, "ǂ": 82, "ǁ": 83,

#     # Implosives
#     "ɓ": 84, "ɗ": 85, "ʄ": 86, "ɠ": 87, "ʛ": 88,

#     # Ejectives
#     "ʼ": 89,

#     # Other Symbols
#     "ɚ": 90, "ɫ": 91, "ɬ": 92, "ɮ": 93, "ʍ": 94, "w": 95, "ɺ": 96,
#     "ɾ̞": 97, "ɕ": 98, "ʑ": 99, "ɧ": 100,

#     # Diacritics (can combine with base symbols)
#     "ˈ": 101, "ˌ": 102, "ː": 103, "ˑ": 104, "̆": 105,
#     "̩": 106, "̯": 107, "ʰ": 108, "ʷ": 109, "ʲ": 110, "˞": 111,
#     "̃": 112, "̴": 113, "̝": 114, "̞": 115, "̘": 116, "̙": 117,
#     "̤": 118, "̰": 119, "̪": 120, "̺": 121, "̻": 122,

#     # Tone and Word Accents
#     "˥": 123, "˦": 124, "˧": 125, "˨": 126, "˩": 127,
#     "˩˥": 128, "˥˩": 129, "˧˥": 130, "˩˧": 131
# }




_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(_vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))

_pad = '_'
_punctuations = '.!;:,?' # list of punctuations that espeak engine will keep after phonemization
_space = ' '

char_list = [_pad] + list(_phonemes) + list(_space) + list(_punctuations) 

phoneme_vocab={s: i for i, s in enumerate(char_list)}


def text_to_phonemes(text: str) -> list:
    phn = phonemize(
        text,
        language='en-us',
        backend='espeak',
        separator=Separator(phone=None, word='', syllable='|'),
        strip=True,
        preserve_punctuation=True,
        njobs=4
    )

    # Split the phonemized text into individual phonemes
    phonemes = phn
    # print(phonemes)
    # print(text)
    print("########")
    # Map phonemes to their corresponding IPA indices
    phoneme_indices = []
    for phoneme in phonemes:
        # Remove any non-IPA characters (if present) such as punctuation or stress marks
        phoneme_cleaned = phoneme.strip("ˈˌ")  # Clean stress marks
        if phoneme_cleaned in phoneme_vocab:
            phoneme_indices.append(phoneme_vocab[phoneme_cleaned])
        else:
            print(f"kunown phonem is  is  {phoneme_cleaned}")
            phoneme_indices.append(1000)  # If phoneme is not found in the vocab

    return phoneme_indices

# # Example usage
# text="The first books were printed in black letter, i.e. the letter which was a Gothic development of the ancient Roman character,"
# text="that the forms of printed letters should follow more or less closely those of the written character, and they followed them very closely."
# phoneme_indices = text_to_phonemes(text)
# print(phoneme_indices)

