import numpy as np
import torch
import random
from PIL import Image


class Subset:
    def __init__(self, data, size):
        assert len(data) >= size, "Sum of support and query samples exceeds the number of samples per class!"
        self.data = data
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        x = [self.data[idx] for idx in np.random.permutation(len(self.data))[:self.size]]
        return x


class Identity:
    def __init__(self):
        pass

    def __call__(self, x):
        return x, 0


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


class Homography:
    """This code is modified from <https://github.com/maple-research-lab/AET>"""
    def __init__(self, shift=4, scale=(0.8, 1.2), fillcolor=255, resample=Image.BILINEAR, normalize=None):
        self.shift = shift
        self.scale = scale
        self.fillcolor = fillcolor
        self.resample = resample
        self.normalize = normalize

    def __call__(self, img):
        width, height = img.size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        shift = [float(random.randint(-int(self.shift), int(self.shift))) for _ in range(8)]
        scale = random.uniform(self.scale[0], self.scale[1])
        rotation = random.randint(0, 3)

        pts = [((0 - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (height - center[1]) * scale + center[1]),
               ((0 - center[0]) * scale + center[0], (height - center[1]) * scale + center[1])]
        pts = [pts[(ii + rotation) % 4] for ii in range(4)]
        pts = [(pts[ii][0] + shift[2 * ii], pts[ii][1] + shift[2 * ii + 1]) for ii in range(4)]

        coeffs = find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )

        kwargs = {"fillcolor": self.fillcolor}
        img2 = img.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

        coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)
        if self.normalize is not None:
            coeffs = self.normalize(coeffs)
        coeffs = coeffs.squeeze()

        return img2, coeffs


class RandomPerspective:
    # Similar to torchvision RandomPerspective but with fillcolor
    def __init__(self, distortion_scale=0.5, interpolation=Image.BICUBIC, fillcolor=0):
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    def __call__(self, img):
        width, height = img.size
        startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
        if random.random() < 0.5:
            startpoints, endpoints = endpoints, startpoints
        coeffs = find_coeffs(endpoints, startpoints)
        return img.transform(img.size, Image.PERSPECTIVE, coeffs, self.interpolation, fillcolor=self.fillcolor)

    @staticmethod
    def get_params(width, height, distortion_scale):
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints
