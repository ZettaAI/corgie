import torch
import numpy as np
from dataclasses import dataclass

class Binarizer:
    def __init__(self, binarization):
        self.bin = binarization
    def __call__(self, tens):
        if self.bin is None:
            return tens
        elif self.bin[0] == 'neq':
            return tens != self.bin[1]
        elif self.bin[0] == 'eq':
            return tens == self.bin[1]
        elif self.bin[0] == 'gt':
            return tens > self.bin[1]
        elif self.bin[0] == 'lt':
            return tens < self.bin[1]

class PartialSpecification:
    def __init__(self, f, **kwargs):
        self.f = f
        self.constr_kwargs = kwargs

    def __call__(self, **kwargs):
        return self.f(**self.constr_kwargs, **kwargs)


@dataclass
class Translation:
    x: float
    y: float

def percentile_trans_adjuster(field, h=25, l=75, unaligned_img=None):
    return  Translation(x=0, y=0)
    if field is None:
        result = Translation(0, 0)
    else:
        nonzero_field_mask = (field[:,0] != 0) & (field[:,1] != 0)

        if unaligned_img is not None:
            no_tissue = field.field().from_pixels()(unaligned_img) == 0
            nonzero_field_mask[..., no_tissue.squeeze()] = False

        nonzero_field = field[..., nonzero_field_mask.squeeze()].squeeze()

        if nonzero_field.sum() == 0:
            result = Translation(0, 0)
        else:
            med_result = Translation(
                    x=int(nonzero_field[0].median()),
                    y=int(nonzero_field[1].median())
                    )

            low_l = percentile(nonzero_field, l)
            high_l = percentile(nonzero_field, h)
            mid = 0.5 * (low_l + high_l)
            result = Translation(x=int(mid[0]), y=int(mid[1]))

    return result

def percentile(field, q):
    # https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    :param field: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (field.shape[1] - 1))
    result = field.kthvalue(k, dim=1).values
    return result

def crop(**kwargs):
    raise NotImplementedError

def expand_to_dims(tens, dims):
    tens_dims = len(tens.shape)
    assert (tens_dims) <= dims
    tens = tens[(None, ) * (dims - tens_dims)]
    return tens

def cast_tensor_type(tens, dtype):
    '''
        tens: pytorch tens
        dtype: string, eg 'float', 'int', 'byte'
    '''
    if dtype is not None:
        assert hasattr(tens, dtype)
        return getattr(tens, dtype)()
    else:
        return tens

def read_mask_list(mask_list, bcube, mip):
    result = None

    for m in mask_list:
        this_data = m.read(bcube=bcube, mip=mip).to(torch.bool)
        if result is None:
            result = this_data
        else:
            result = result | this_data

    return result

def crop(data, c):
    if c == 0:
        return data
    else:
        if data.shape[-1] == data.shape[-2]:
            return data[..., c:-c, c:-c]
        elif data.shape[-2] == data.shape[-3] and data.shape[-1] == 2: #field
            return data[..., c:-c, c:-c, :]

def coarsen_mask(mask, n=1, flip=False):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kernel_var = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(mask.device).float()
    k = torch.nn.Parameter(data=kernel_var, requires_grad=False)
    for _ in range(n):
        if flip:
            mask = mask.logical_not()
        mask =  (torch.nn.functional.conv2d(mask.float(),
                                kernel_var, padding=1) > 1)
        if flip:
            mask = mask.logical_not()
        mask = mask

    return mask


