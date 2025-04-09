import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure, filters


def main():

    refs = np.zeros((768, 7*768), np.float32)
    path = 'data/imax/original/images'
    files = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png']
    for i, imax in enumerate(files):
        ref = np.array(Image.open(os.path.join(
            path, imax)))
        refs[:, i*768:(i+1)*768] = ref
    reference = refs

    for path in os.listdir('data/dkist/images'):
        arr = np.load(f'data/dkist/images/{path}')['X']
        arr = arr[321:-321, 321:-321]
        img = Image.fromarray(arr)
        img = img.resize((768, 768))
        x = np.array(img, dtype=np.float32)

        reference_sharpness = np.var(filters.laplace(reference))
        target_sharpness = np.var(filters.laplace(x))
        sigma = np.sqrt(target_sharpness / reference_sharpness)

        matched = filters.gaussian(x, sigma=sigma)
        matched = exposure.match_histograms(matched, reference)

        matched /= np.max(matched)
        plt.imshow(matched, cmap='gray')
        plt.show()

        np.savez_compressed(
            f'data/dkist/transformed/{path[:-3]}npz', X=matched)


if __name__ == '__main__':
    main()
