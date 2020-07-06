import click
import procspec

from corgie import scheduling, argparsers, helpers

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

from corgie.cli.common     import ChunkedJob
from corgie.cli.downsample import DownsampleJob


class ComputeFieldJob(scheduling.Job):
    def __init__(self, src_stack, tgt_stack, dst_layer,
            chunk_xy, chunk_z, processor_spec, processor_mip,
            pad, crop,
            bcube, tgt_z_offset, blend_xy=0, suffix=''):

        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.blend_xy = blend_xy
        self.pad = pad
        self.crop = crop
        self.bcube = bcube
        self.tgt_z_offset = tgt_z_offset
        self.suffix = suffix #in case this job wants to make more layers
        self.processor_spec = processor_spec
        self.processor_mip = processor_mip
        if isinstance(self.processor_spec, str):
            self.processor_spec = [self.processor_spec]
        if isinstance(self.processor_mip, int):
            self.processor_mip = [self.processor_mip]

        if len(self.processor_mip) != len(self.processor_spec):
            raise exceptions.CorgieException("The number of processors doesn't "
                    "match the number of specified processor MIPs")

        super().__init__()

    def task_generator(self):
        for i in range(len(self.processor_spec)):
            this_proc = self.processor_spec[i]
            this_proc_mip = self.processor_mip[i]
            is_last_proc = i == len(self.processor_spec) - 1

            this_task = helpers.PartialSpecification(
                    ComputeFieldTask,
                    src_stack=self.src_stack,
                    tgt_stack=self.tgt_stack,
                    processor_spec=this_proc,
                    pad=self.pad,
                    crop=self.crop,
                    tgt_z_offset=self.tgt_z_offset
                    )

            chunked_job = ChunkedJob(
                    task_class=this_task,
                    dst_layer=self.dst_layer,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    blend_xy=self.blend_xy,
                    mip=this_proc_mip,
                    bcube=self.bcube,
                    suffix=self.suffix
                    )

            yield from chunked_job.task_generator

            # this processors MIP has the freshest field

            if not is_last_proc:
                yield scheduling.wait_until_done
                self.dst_layer.data_mip = this_proc_mip

                next_proc_mip = self.processor_mip[i + 1]
                if this_proc_mip < next_proc_mip:
                    downsample_job = DownsampleJob(
                                src_layer=self.dst_layer,
                                chunk_xy=self.chunk_xy,
                                chunk_z=self.chunk_z,
                                mip_start=this_proc_mip,
                                mip_end=next_proc_mip,
                                bcube=self.bcube
                                )
                    yield from downsample_job.task_generator
                    yield scheduling.wait_until_done


        if self.processor_mip[0] > self.processor_mip[-1]:
            # good manners
            # prepare the ground for the next you
            downsample_job = DownsampleJob(
                            src_layer=self.dst_layer,
                            chunk_xy=self.chunk_xy,
                            chunk_z=self.chunk_z,
                            mip_start=self.processor_mip[-1],
                            mip_end=self.processor_mip[0],
                            bcube=self.bcube
                            )

            yield scheduling.wait_until_done
            yield from downsample_job.task_generator

            # field is fresh at all mip layers
            self.dst_layer.data_mip = None


class ComputeFieldTask(scheduling.Task):
    def __init__(self, processor_spec, src_stack, tgt_stack, dst_layer,  mip,
            pad, crop, tgt_z_offset, bcube):
        super().__init__()
        self.processor_spec = processor_spec
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.crop = crop
        self.tgt_z_offset = tgt_z_offset
        self.bcube = bcube

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.mip)
        tgt_bcube = src_bcube.translate(z_offset=self.tgt_z_offset)

        processor = procspec.parse_proc(
                spec_str=self.processor_spec)
        src_translation, src_data_dict = self.src_stack.read_data_dict(src_bcube,
                mip=self.mip, stack_name='src')
        tgt_translation, tgt_data_dict = self.tgt_stack.read_data_dict(tgt_bcube,
                mip=self.mip, stack_name='tgt')

        processor_input = {**src_data_dict, **tgt_data_dict}

        predicted_field = processor(processor_input,
                                    output_key='src_cf_field')

        cropped_field = helpers.crop(predicted_field, self.crop)
        cropped_field[:, 0] += src_translation.x
        cropped_field[:, 1] += src_translation.y
        self.dst_layer.write(cropped_field, bcube=self.bcube, mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--tgt_layer_spec', '-t', nargs=1,
        type=str, required=False, multiple=True,
        help='Target layer spec. Use multiple times to include all masks, fields, images. '\
                'DEFAULT: Same as source layers')

@corgie_option('--dst_layer_spec',  nargs=1,
        type=str, required=False,
        help= "Specification for the destination layer. Must be a field type." + \
                " DEFAULT: source reference key path + /field/cf_field + (_{suffix})?")

@corgie_option('--reference_key',        nargs=1, type=str, default='img')

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--blend_xy',             nargs=1, type=int, default=0)
@corgie_option('--pad',                  nargs=1, type=int, default=512,
        )
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--processor_spec',       nargs=1, type=str, multiple=True,
        required=True)
@corgie_option('--processor_mip',                  nargs=1, type=int, multiple=True,
        required=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@corgie_option('--tgt_z_offset',     nargs=1, type=str, default=1)

@click.option('--suffix',            nargs=1, type=str,  default=None)

@click.pass_context
def compute_field(ctx, src_layer_spec, tgt_layer_spec, dst_layer_spec,
        suffix, processor_spec, pad, crop, chunk_xy, start_coord, processor_mip,
        end_coord, coord_mip, blend_xy, tgt_z_offset, chunk_z, reference_key):
    if suffix is None:
        suffix = ''
    else:
        suffix = f"_{suffix}"

    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    tgt_stack = create_stack_from_spec(tgt_layer_spec,
            name='tgt', readonly=True, reference=src_stack)

    reference_layer = None
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]
    dst_layer = create_layer_from_spec(dst_layer_spec, allowed_types=['field'],
            default_type='field', readonly=False, caller_name='dst_layer',
            reference=reference_layer, overwrite=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    compute_field_job = ComputeFieldJob(
            src_stack=src_stack,
            tgt_stack=tgt_stack,
            dst_layer=dst_layer,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            blend_xy=blend_xy,
            processor_spec=processor_spec,
            pad=pad,
            crop=crop,
            bcube=bcube,
            tgt_z_offset=tgt_z_offset,
            suffix=suffix,
            processor_mip=processor_mip)

    # create scheduler and execute the job
    scheduler.register_job(compute_field_job, job_name="Compute field {}, tgt z offset {}".format(
        bcube, tgt_z_offset))
    scheduler.execute_until_completion()
