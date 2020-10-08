import click

import torch
import torchfields
from copy import deepcopy

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option


class InterpolateJob(scheduling.Job):
    def __init__(self, 
                 src_layer, 
                 dst_layer, 
                 src_mip, 
                 dst_mip, 
                 factor,
                 bcube, 
                 chunk_xy):
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.src_mip = src_mip
        self.dst_mip = dst_mip
        self.factor = factor
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=1,
                mip=self.dst_mip)
        tasks = [InterpolateTask(src_layer=self.src_layer,
                                dst_layer=self.dst_layer,
                                src_mip=self.src_mip,
                                dst_mip=self.dst_mip,
                                factor=self.factor,
                                bcube=input_chunk) for input_chunk in chunks]
        print("Yielding interpolation tasks for bcube: {}".format(self.bcube))
        yield tasks

class InterpolateTask(scheduling.Task):
    def __init__(self, 
                 src_layer, 
                 dst_layer, 
                 src_mip, 
                 dst_mip, 
                 factor,
                 bcube):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.src_mip = src_mip
        self.dst_mip = dst_mip
        self.factor = factor
        self.pad = 1,
        self.crop = 1,
        self.bcube = bcube

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.src_mip)
        src_bcube = self.bcube.scale(self.factor)
        src_data = self.src_layer.read(bcube=src_bcube, mip=self.src_mip)
        # How to scale depends on layer type.
        # Images are avg pooled, masks are max pooled, segmentation is...
        interpolator = self.src_layer.get_interpolator(factor=self.factor)
        dst_data = interpolator(src_data)
        cropped_data = helpers.crop(dst_data, self.crop)
        self.dst_layer.write(cropped_data, bcube=self.bcube, mip=self.dst_mip)
        src_data = dst_data


@click.command()
@corgie_optgroup('Layer Parameterd')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True,
        help='Specification for the source layer. ' + \
                LAYER_HELP_STR)

@corgie_option('--dst_layer_spec',  '-s', nargs=1,
        type=str, required=True,
        help= "Specification for the destination layer. "+ \
                "Refer to 'src_layer_spec' for parameter format." + \
                " DEFAULT: Same as src_layer_spec")

@corgie_optgroup('Interpolate parameters')
@corgie_option('--scale_factor',  '-f', nargs=1, type=float, required=True)
@corgie_option('--src_mip',           , nargs=1, type=int, default=0)
@corgie_option('--chunk_xy',      '-c', nargs=1, type=int, default=2048)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@click.pass_context
def scale(ctx, 
          src_layer_spec, 
          dst_layer_spec, 
          src_mip,
          scale_factor,
          chunk_xy,
          start_coord,
          end_coord,
          coord_mip):
    scheduler = ctx.obj['scheduler']
    corgie_logger.debug("Setting up Source and Destination layers...")

    src_layer = create_layer_from_spec(src_layer_spec,
            readonly=True)
    info = deepcopy(src_layer.get_info())
    # TODO: check where in the MIP stack this image belongs
    # TODO: allow for writing to source
    highest_mip = deepcopy(info['scales'][src_mip])
    highest_mip_info['size'] = [
            ceil(highest_mip_info['size'][0] * scale_factor),
            ceil(highest_mip_info['size'][1] * scale_factor),
            highest_mip_info['size'][2]
        ]

    highest_mip_info['voxel_offset'] = [
            ceil(highest_mip_info['voxel_offset'][0] * scale_factor),
            ceil(highest_mip_info['voxel_offset'][1] * scale_factor),
            highest_mip_info['voxel_offset'][2]
        ]
    # TODO: Account for fractional resolutions with CloudVolume update
    highest_mip_info['resolution'] = [
            ceil(highest_mip_info['resolution'][0] * scale_factor),
            ceil(highest_mip_info['resolution'][1] * scale_factor),
            highest_mip_info['resolution'][2]
        ]
    info['scales'] = [highest_mip] 
    # dummy object for get_info() method
    dst_ref = MiplessCloudVolume(path='file://tmp/cloudvolume/empty',
                                 info=info,
                                 overwrite=False)
    dst_layer = create_layer_from_spec(dst_layer_spec,
        readonly=False,
        reference=dst_ref, chunk_z=chunk_z, overwrite=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    interpolation_job = InterpolateJob(src_layer=src_layer, 
                               dst_layer=dst_layer,
                               src_mip=src_mip,
                               dst_mip=0,
                               factor=factor,
                               bcube=bcube,
                               chunk_xy=chunk_xy)

    # create scheduler and execute the job
    scheduler.register_job(interpolation_job, job_name="interpolation")
    scheduler.execute_until_completion()
    result_report = f"Interpolated {src_layer} by {factor}. Result in {dst_layer}"
    corgie_logger.info(result_report)
