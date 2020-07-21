import functools
import click
from scipy import ndimage
import numpy as np

from corgie import scheduling, helpers

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option


class MorphologyJob(scheduling.Job):
    def __init__(self, src_layer, dst_layer, mip, pad, bcube, chunk_xy, chunk_z,
                 *,
                 mask_layer=None, op=None, kernel=None, radius=1,
                 iterations=1, origin=(0,0), border_value=0
                 ):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mask_layer = mask_layer
        self.op = op
        self.kernel = kernel
        self.radius = radius
        self.iterations = iterations
        self.origin = origin
        self.border_value = border_value

    def task_generator(self):
        kwargs = {
            'src_layer': self.src_layer,
            'dst_layer': self.dst_layer,
            'mip': self.mip,
            'pad': self.pad,
            'mask_layer': self.mask_layer,
            'kernel': self.kernel,
            'radius': self.radius,
            'iterations': self.iterations,
            'origin': self.origin,
            'border_value': self.border_value,
        }

        chunks = self.dst_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip
                )

        if self.op == "binary_closing":
            tasks = (
                BinaryClosingTask(bcube=input_chunk, **kwargs)
                for input_chunk in chunks
            )
        elif self.op == "binary_dilation":
            tasks = (
                BinaryDilationTask(bcube=input_chunk, **kwargs)
                for input_chunk in chunks
            )
        elif self.op == "binary_erosion":
            tasks = (
                BinaryErosionTask(bcube=input_chunk, **kwargs)
                for input_chunk in chunks
            )
        elif self.op == "binary_opening":
            tasks = (
                BinaryOpeningTask(bcube=input_chunk, **kwargs)
                for input_chunk in chunks
            )
        else:
            raise NotImplementedError(f"{self.op} task does not exist")

        corgie_logger.info(f"Yielding {self.op} tasks for bcube: {self.bcube}, MIP: {self.mip}")
        yield list(tasks)

class MorphologyTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel: str, radius, iterations, origin,
                 border_value):
        super().__init__(self)
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.mask_layer = mask_layer
        self.kernel = kernel.lower()
        self.radius = radius
        self.iterations = iterations
        self.origin = origin
        self.border_value = border_value

    @property
    @functools.lru_cache()
    def structure(self):
        N = 2 * self.radius + 1
        y_off = self.origin[0] + self.radius
        x_off = self.origin[1] + self.radius
        y, x = np.ogrid[-y_off : N - y_off, -x_off : N - x_off]

        if self.kernel == 'disk':
            return x * x + y * y <= self.radius * self.radius
        elif self.kernel == 'diamond':
            return abs(x) + abs(y) <= self.radius
        elif self.kernel == 'box':
            return np.maximum(abs(x), abs(y)) <= self.radius


    def execute(self, morph_op):
        padded_bcube = self.bcube.uncrop(self.pad, self.mip)

        src_data = self.src_layer.read(bcube=padded_bcube, mip=self.mip).numpy()
        src_data = np.transpose(src_data, (2,3,0,1)).squeeze()

        mask_data = None
        if self.mask_layer is not None:
            mask_data = self.mask_layer.read(bcube=padded_bcube, mip=self.mip).numpy()
            mask_data = np.transpose(mask_data, (2,3,0,1)).squeeze()

        dst_data = morph_op(
            src_data,
            structure=self.structure,
            iterations=self.iterations,
            origin=self.origin,
            mask=mask_data,
            border_value=self.border_value,
        )

        dst_data = np.expand_dims(np.atleast_3d(dst_data), 3)
        dst_data = np.transpose(dst_data, (2,3,0,1))

        cropped_out = helpers.crop(dst_data, self.pad)
        self.dst_layer.write(cropped_out, bcube=self.bcube, mip=self.mip)
        print(f"Done with {self.bcube}")

class BinaryClosingTask(MorphologyTask):
    def __init__(self, src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value):
        super().__init__(src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value)

    def execute(self):
        super().execute(ndimage.binary_closing)

class BinaryDilationTask(MorphologyTask):
    def __init__(self, src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value):
        super().__init__(src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value)

    def execute(self):
        super().execute(ndimage.binary_dilation)

class BinaryErosionTask(MorphologyTask):
    def __init__(self, src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value):
        super().__init__(src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value)

    def execute(self):
        super().execute(ndimage.binary_erosion)

class BinaryOpeningTask(MorphologyTask):
    def __init__(self, src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value):
        super().__init__(src_layer, dst_layer, mip, pad, mask_layer,
                 bcube, kernel, radius, iterations, origin,
                 border_value)

    def execute(self):
        super().execute(ndimage.binary_opening)


@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)

@corgie_option('--dst_layer_spec',  nargs=1,
        type=str, required=True,
        help= "Destination layer spec. where output layer will go")

@corgie_optgroup('Morphological Operation Parameters')
@corgie_option('--op',                   nargs=1, type=str, required=True)
@corgie_option('--mask_spec',            nargs=1, type=str,
        help='Optional input mask layer spec. If a mask is given, only those elements with a True value at the corresponding mask element are modified at each iteration.')
@corgie_option('--kernel',               nargs=1, type=click.Choice(['diamond', 'box', 'disk'], case_sensitive=False))
@corgie_option('--radius',               nargs=1, type=int, default=1)
@corgie_option('--iterations',           nargs=1, type=int, default=1)
@corgie_option('--origin',               nargs=2, type=int, default=(0,0))
@corgie_option('--border_value',         nargs=1, type=int, default=0)

@corgie_optgroup('Task Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, required=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def morphological_op(ctx, src_layer_spec, dst_layer_spec, op, pad, mip,
         mask_spec, kernel, radius, iterations, origin, border_value, 
         chunk_xy, chunk_z, start_coord, end_coord, coord_mip):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = create_layer_from_spec(src_layer_spec,
            caller_name='src layer',
            readonly=True)

    mask_layer = None
    if mask_spec is not None:
        mask_layer = create_layer_from_spec(mask_spec,
                caller_name='src mask',
                readonly=True,
                allowed_types=['mask'],
                default_type='mask')

    dst_layer = create_layer_from_spec(dst_layer_spec, reference=src_layer,
            caller_name="dst layer", allowed_type=["img"], default_type="img", readonly=False)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    morphology_job = MorphologyJob(
        src_layer=src_layer,
        dst_layer=dst_layer,
        mip=mip,
        pad=pad,
        bcube=bcube,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        mask_layer=mask_layer,
        op=op,
        kernel=kernel,
        radius=radius,
        iterations=iterations,
        origin=origin,
        border_value=border_value,
    )

    # create scheduler and execute the job
    scheduler.register_job(morphology_job, job_name="Morphological {} Operation {}".format(op, bcube))
    scheduler.execute_until_completion()
