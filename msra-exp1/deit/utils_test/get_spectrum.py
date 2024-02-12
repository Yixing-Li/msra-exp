import numpy as np
import math
from scipy.linalg import dft
import matplotlib.pyplot as plt

len_tokens = 197

F = dft(len_tokens, scale='sqrtn')

def calc_spectrum(A):
    return F @ A @ F.T

def plot_spectrum(a):
  s = calc_spectrum(a)
  s = np.linalg.norm(s, ord=2, axis=1)
  s = np.concatenate([s[-math.floor(len_tokens/2):], s[0:1], s[1:math.floor(len_tokens/2)]], axis=0)
  plt.plot(s)