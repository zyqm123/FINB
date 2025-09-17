import math
import numbers
import random
from typing import Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


class RandomMixup(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, data) -> Tuple[Tensor, Tensor]:
        batch = data['img']
        target = data['label']

        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return {'img': batch, 'label': target}

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return {'img': batch, 'label': target}

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


class RandomCutmix(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, data) -> Tuple[Tensor, Tensor]:
        batch = data['img']
        target = data['label']

        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return {'img': batch, 'label': target}

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return {'img': batch, 'label': target}

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


class RandomSwap(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return self.swap(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def swap(self, img, crop):
        def crop_image(image, cropnum):
            width, high = image.size
            crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
            crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(
                        image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
            return im_list

        widthcut, highcut = img.size
        img = img.crop((10, 10, widthcut - 10, highcut - 10))
        images = crop_image(img, crop)
        pro = 5
        if pro >= 5:
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)

            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i = i.resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        return toImage