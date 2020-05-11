import os
import copy
import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Bbox

from torch.nn.functional import interpolate

from corgie import layers

from corgie.log import logger as corgie_logger

from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends.base import DataBackendBase, BaseLayerBackend, \
        register_backend

@register_backend("cv")
class CVDataBackend(DataBackendBase):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class CVLayerBase(BaseLayerBackend):
    def __init__(self, path, backend, reference=None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.cv = MiplessCloudVolume(path)
        self.backend = backend

        try:
            self.cv.get_info()
        except cv.exceptions.InfoUnavailableError as e:
            if reference is None:
                raise e
            else:
                info = copy.deepcopy(reference.get_info())

                info['num_channels'] = self.get_num_channels()
                info['data_type'] = self.get_data_type()
                if not self.supports_voxel_offset():
                    for scale in info['scales']:
                        scale['voxel_offset'] = [0, 0, 0]
                if not self.supports_chunking():
                    for scale in info['scales']:
                        scale['chunk_sizes'] = [[1, 1, 1]]

                self.cv = MiplessCloudVolume(path,
                        info=info)

    def __str__(self):
        return "CV {}".format(self.path)

    def get_sublayer(self, name, layer_type, **kwargs):
        sublayer_path = os.path.join(self.cv.path, layer_type, name)
        return self.backend.create_layer(path=sublayer_path, layer_type=layer_type,
                reference=self, **kwargs)

    def read_backend(self, bcube, mip,
            channel_start=None, channel_end=None):
        if channel_start is None:
            channel_start = 0
        if channel_end is None:
            channel_end = self.cv[mip].shape[-1]

        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        corgie_logger.debug("READ from {}: \n   x: {}, y: {}, z: {}, chan: {}, MIP: {}".format(
            str(self), x_range, y_range, z_range, (channel_start, channel_end), mip))
        data = self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1],
                     channel_start:channel_end]
        data = np.transpose(data, (2,3,0,1))
        return data

    def write_backend(self, data, bcube, mip,
            channel_start=None, channel_end=None):
        if channel_start is None:
            channel_start = 0
        if channel_end is None:
            channel_end = self.cv[mip].shape[-1]

        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        data = np.transpose(data, (2,3,0,1))

        corgie_logger.debug("Write to {}: \n x: {}, y: {}, z: {}, chan: {}, MIP: {}".format(
            str(self), x_range, y_range, z_range, (channel_start, channel_end), mip))
        import pdb; pdb.set_trace()
        self.cv[mip].autocrop = True
        self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1],
                     channel_start:channel_end] = data
        self.cv[mip].autocrop = False
        test_data = self.read_backend(bcube, mip, channel_start, channel_end)


    def get_info(self):
        return self.cv.get_info()

    def declare_write_region(self, bcube, mips):
        for m in mips:
            self.cv.ensure_info_has_mip(m)
        aligned_bcube = self.get_chunk_aligned_bcube(bcube, max(mips))
        super().declare_write_region(aligned_bcube, mips)

    def get_chunk_aligned_bcube(self, bcube, mip):
        bbox = Bbox((bcube.x_range(mip)[0], bcube.y_range(mip)[0], bcube.z_range()[0]),
                    (bcube.x_range(mip)[1], bcube.y_range(mip)[1], bcube.z_range()[1]))

        aligned_bbox = bbox.expand_to_chunk_size(self.cv[mip].chunk_size,
                                                 self.cv[mip].voxel_offset)

        aligned_bcube = copy.deepcopy(bcube)
        aligned_bcube.reset_coords(aligned_bbox.minpt[0], aligned_bbox.maxpt[0],
                                   aligned_bbox.minpt[1], aligned_bbox.maxpt[1],
                                   aligned_bbox.minpt[2], aligned_bbox.maxpt[2],
                                   mip=mip)
        return aligned_bcube

    def break_bcube_into_chunks(self, bcube, chunk_xy, chunk_z, mip,
            readonly=False):
        if not readonly:
            bcube = self.get_chunk_aligned_bcube(bcube, mip)
        chunks = super().break_bcube_into_chunks(bcube, chunk_xy, chunk_z,
                mip)
        return chunks


@CVDataBackend.register_layer_type_backend("img")
class CVImgLayer(CVLayerBase, layers.ImgLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("field")
class CVFieldLayer(CVLayerBase, layers.FieldLayer):
    backend_types = ['float32', 'int16']
    def __init__(self, backend_dtype='float32', *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        if backend_dtype not in self.supported_backend_dtypes:
            raise exceptions.ArgumentError("Field layer 'backend_type'",
                    "\n{} is not a supported field backend data type. \n"
                    "Supported backend data types: {}".format(backend_type,
                        self.supported_backend_dtypes)
                    )
        self.backend_dtype = backend_dtype


@CVDataBackend.register_layer_type_backend("mask")
class CVMaskLayer(CVLayerBase, layers.MaskLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("section_value")
class CVSectionValueLayer(CVLayerBase, layers.SectionValueLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)