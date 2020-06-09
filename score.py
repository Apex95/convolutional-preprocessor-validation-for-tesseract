import pytesseract
import Levenshtein
from PIL import Image
import matplotlib.pyplot as plt

import conv


def run_through_tesseract(img):
  
  img = Image.fromarray(img.squeeze(), 'L' if conv.n_of_channels == 1 else 'RGB')
  tess_output = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

  recognized_text = ' '.join(s for s in tess_output["text"] if len(s) > 0) 
  
  return recognized_text


def compute_cer(recognized_text, expected_text):
  return Levenshtein.distance(recognized_text, expected_text) / len(expected_text)

def compute_lcser(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
 
    # read a substring from the matrix
    result = ''
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result += a[i-1]
 
    return len(a)-len(result) + len(b)-len(result)


def compute_wer(recognized_text, expected_text):
  word_set = set(recognized_text.split() + expected_text.split())
  word2char = dict(zip(word_set, range(len(word_set))))

  word1 = [chr(word2char[w]) for w in recognized_text.split()]
  word2 = [chr(word2char[w]) for w in expected_text.split()]

  return Levenshtein.distance(''.join(word1), ''.join(word2)) / len(word2)

def get_lcs(a, b):
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
 
    result = ''
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result += a[i-1]

    return result

def compute_precision(recognized_text, expected_text):
  return 0 if len(recognized_text) == 0 else len(get_lcs(recognized_text, expected_text)) / len(recognized_text)

def compute_recall(recognized_text, expected_text):
  return len(get_lcs(recognized_text, expected_text)) / len(expected_text)



def compute_score(img, expected_text):

  recognized_text = run_through_tesseract(img)
  
  return compute_cer(recognized_text, expected_text), compute_wer(recognized_text, expected_text), compute_lcser(recognized_text, expected_text), compute_precision(recognized_text, expected_text), compute_recall(recognized_text, expected_text), True if recognized_text == expected_text else False


