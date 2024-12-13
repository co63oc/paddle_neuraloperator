from typing import List  # noqa

import paddle

from ...training.patching import MultigridPatching2D
from .base_transforms import Transform


class MGPatchingTransform(Transform):
    def __init__(
        self,
        model: paddle.nn.Layer,
        levels: int,
        padding_fraction: float,
        stitching: float,
    ):
        """Wraps MultigridPatching2D to expose canonical
        transform .transform() and .inverse_transform() API

        Parameters
        ----------
        model: nn.Layer
            model to wrap in MultigridPatching2D
        levels : int
            mg_patching level parameter for MultigridPatching2D
        padding_fraction : float
            mg_padding_fraction parameter for MultigridPatching2D
        stitching : float
            mg_patching_stitching parameter for MultigridPatching2D
        """
        super.__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.patcher = MultigridPatching2D(
            model=model,
            levels=self.levels,
            padding_fraction=self.padding_fraction,
            stitching=self.stitching,
        )

    def transform(self, data_dict):
        x = data_dict["x"]
        y = data_dict["y"]
        x, y = self.patcher.patch(x, y)
        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict

    def inverse_transform(self, data_dict):
        x = data_dict["x"]
        y = data_dict["y"]
        x, y = self.patcher.unpatch(x, y)
        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict

    def to(self, _):
        return self


class RandomMGPatch:
    def __init__(self, levels=2):
        self.levels = levels
        self.step = 2**levels

    def __call__(self, data):
        def _get_patches(shifted_image, step, height, width):
            """Take as input an image and return multi-grid patches centered around the middle of the image"""
            if step == 1:
                return (shifted_image,)
            else:
                # Notice that we need to stat cropping at start_h = (height - patch_size)//2
                # (//2 as we pad both sides)
                # Here, the extracted patch-size is half the size so patch-size = height//2
                # Hence the values height//4 and width // 4
                start_h = height // 4
                start_w = width // 4
                patches = _get_patches(
                    shifted_image[:, start_h:-start_h, start_w:-start_w],
                    step // 2,
                    height // 2,
                    width // 2,
                )
                return shifted_image[:, ::step, ::step], *patches

        x, y = data
        channels, height, width = tuple(x.shape)
        center_h = height // 2
        center_w = width // 2

        # Sample a random patching position
        pos_h = paddle.randint(low=0, high=height, shape=(1,))[0]
        pos_w = paddle.randint(low=0, high=width, shape=(1,))[0]
        shift_h = center_h - pos_h
        shift_w = center_w - pos_w

        shifted_x = paddle.roll(x=x, shifts=(shift_h, shift_w), axis=(0, 1))
        patches_x = _get_patches(shifted_x, self.step, height, width)
        shifted_y = paddle.roll(x=y, shifts=(shift_h, shift_w), axis=(0, 1))
        patches_y = _get_patches(shifted_y, self.step, height, width)
        return paddle.concat(x=patches_x, axis=0), patches_y[-1]


class MGPTensorDataset(paddle.io.Dataset):
    def __init__(self, x, y, levels=2):
        assert x.shape[0] == y.shape[0], "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.levels = 2
        self.transform = RandomMGPatch(levels=levels)

    def __getitem__(self, index):
        return self.transform((self.x[index], self.y[index]))

    def __len__(self):
        return self.x.shape[0]
