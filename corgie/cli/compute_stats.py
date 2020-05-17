import math

import click
from click_option_group import optgroup

from corgie import scheduling
from corgie import helpers
from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option


class ComputeStatsJob(scheduling.Job):
    def __init__(self, src_layer, mask_layers, mean_layer, var_layer, bcube,
            mip, chunk_xy, chunk_z):
        self.src_layer = src_layer
        self.mask_layers = mask_layers
        self.mean_layer = mean_layer
        self.var_layer = var_layer
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

        super().__init__(self)

    def task_generator(self):
        chunks = self.src_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip)

        assert len(chunks) % self.bcube.z_size() == 0
        chunks_per_section = len(chunks) // self.bcube.z_size()

        chunk_mean_layer  = self.mean_layer.get_sublayer(
                name="chunk_mean",
                layer_type="section_value",
                num_channels=chunks_per_section)

        chunk_var_layer  = self.var_layer.get_sublayer(
                name="chunk_var",
                layer_type="section_value",
                num_channels=chunks_per_section)

        chunks = self.src_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip)

        for l in [self.mean_layer, chunk_mean_layer, self.var_layer,
                chunk_var_layer]:
            l.declare_write_region(self.bcube, mips=[self.mip])
        # sort chunks by z
        chunks.sort(reverse=True, key=lambda c:c.z_range()[1])

        tasks = [ComputeStatsTask(self.src_layer,
                                mean_layer=chunk_mean_layer,
                                mask_layers=self.mask_layers,
                                var_layer=chunk_var_layer,
                                mip=self.mip,
                                bcube=chunks[chunk_num],
                                # chunks are sorted by z, so this gives the chunk num
                                # for a given z
                                write_channel=chunk_num % chunks_per_section) \
                            for chunk_num in range(len(chunks))]

        corgie_logger.info("Yielding chunk stats tasks: {}, MIP: {}".format(
            self.bcube, self.mip))
        yield tasks
        yield scheduling.wait_until_done

        accum_chunks = chunk_mean_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=chunks_per_section,
                chunk_z=self.chunk_z,
                mip=self.mip)

        accum_mean_tasks = [ComputeStatsTask(chunk_mean_layer,
                                mean_layer=self.mean_layer,
                                var_layer=None,
                                mip=self.mip,
                                bcube=accum_chunk,
                                write_channel=0) \
                            for accum_chunk in accum_chunks]

        accum_var_tasks = [ComputeStatsTask(chunk_var_layer,
                                mean_layer=self.var_layer,
                                var_layer=None,
                                mip=self.mip,
                                bcube=accum_chunk,
                                write_channel=0) \
                            for accum_chunk in accum_chunks]

        corgie_logger.info("Yielding chunk stats aggregation tasks...")
        yield accum_mean_tasks + accum_var_tasks

    def create_dst_layers(self):
        return mean_layer, var_layer


class ComputeStatsTask(scheduling.Task):
    def __init__(self, src_layer, mean_layer, var_layer, mip,
                 bcube, write_channel, mask_layers=[]):
        super().__init__(self)
        self.src_layer = src_layer
        self.mask_layers = mask_layers
        self.mean_layer = mean_layer
        self.var_layer = var_layer
        self.mip = mip
        self.bcube = bcube
        self.write_channel = write_channel

    def execute(self):
        src_data = self.src_layer.read(bcube=self.bcube,
                mip=self.mip)
        mask_data = helpers.read_mask_list(
                mask_list=self.mask_layers,
                bcube=self.bcube, mip=self.mip)

        if mask_data is not None:
            src_data = src_data[mask_data == 0]

        if self.mean_layer is not None:
            mean = src_data[src_data != 0].float().mean()
            self.mean_layer.write(
                    mean,
                    bcube=self.bcube,
                    mip=self.mip,
                    channel_start=self.write_channel,
                    channel_end=self.write_channel + 1)

        if self.var_layer is not None:
            var = src_data[src_data != 0].float().var()

            self.var_layer.write(
                    var,
                    bcube=self.bcube,
                    mip=self.mip,
                    channel_start=self.write_channel,
                    channel_end=self.write_channel + 1)


@click.command()
@corgie_optgroup('Layer parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)

@corgie_option('--dst_folder',  nargs=1, type=str, required=True,
        help="Folder where aligned stack will go")


# Other Params
@corgie_option('--suffix',     '-s', nargs=1, type=str, default=None)

@corgie_optgroup('Computation Specification')
@corgie_option('--mip',        '-m', nargs=1, type=int, required=True)
@corgie_option('--chunk_xy',   '-c', nargs=1, type=int, default=4096)
@corgie_option('--chunk_z',          nargs=1, type=int, default=1)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@click.pass_context
def compute_stats(ctx, src_layer_spec, dst_folder, suffix, mip,
        chunk_xy, chunk_z,  start_coord, end_coord, coord_mip):
    compute_stats_fn(ctx, src_layer_spec, mask_layers_spec, suffix, mip,
        chunk_xy, chunk_z,  start_coord, end_coord, coord_mip)


def compute_stats_fn(ctx, src_layer_spec, dst_folder, suffix, mip,
        chunk_xy, chunk_z,  start_coord, end_coord, coord_mip):

    if chunk_z != 1:
        raise NotImplemented("Compute Statistics command currently only \
                supports per-section statistics.")

    scheduler = ctx.obj['scheduler']

    src_layer = create_layer_from_spec(src_layer_spec,
            caller_name='src layer',
            readonly=True)

    mask_layers = [create_layer_from_spec(mask_spec,
            caller_name='src mask',
            readonly=True,
            allowed_types=['mask'],
            default_type='mask') for mask_spec in mask_layers_spec
    ]

    if suffix is None:
        suffix = ''
    else:
        suffix = '_' + suffix

    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    mask_layers = src_stack.get_layers_of_type(["mask"])
    non_mask_layers = src_stack.get_layers_of_type(["img", "field"])

    for l in non_mask_layers:
        mean_layer = src_layer.get_sublayer(
                name=f"mean{suffix}",
                path=os.path.join(dst_dir, f"mean{suffix}"),
                layer_type="section_value",
                )

        var_layer  = src_layer.get_sublayer(
                path=os.path.join(dst_dir, f"var{suffix}"),
                layer_type="section_value")


        compute_stats_job = ComputeStatsJob(
               src_layer=l,
               mask_layers=mask_layers,
               mean_layer=mean_layer,
               var_layer=var_layer,
               bcube=bcube,
               mip=mip,
               chunk_xy=chunk_xy,
               chunk_z=chunk_z)

        # create scheduler and execute the job
        scheduler.register_job(compute_stats_job, job_name=f"Compute Stats. Layer: {l}, Bcube: {bcube}")
    scheduler.execute_until_completion()



