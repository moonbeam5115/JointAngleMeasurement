import numpy as np
import matplotlib.pyplot as plt

def plots(ims, figsize=(24, 12), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims - np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((1, 2, 3, 0))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) %2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i]/255, interpolation=None if interp else 'none')
