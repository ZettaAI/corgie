import json
import click

from click_option_group import optgroup

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, str_to_layer_type, \
        DEFAULT_LAYER_TYPE
from corgie.data_backends import get_data_backends, str_to_backend, \
        DEFAULT_DATA_BACKEND
from corgie import exceptions


corgie_optgroup = optgroup.group

def corgie_option(*args, **kwargs):
    return optgroup.option(*args, show_default=True, **kwargs)


def backend_argument(name):
    def wrapper(f):
        backend_args = corgie_option('--{}_backend_args'.format(name),
                nargs=1,
                type=str,
                default='{}',
                help="JSON string describing additional backend args")
        backend = corgie_option('--{}_backend'.format(name),
                type=click.Choice(get_data_backends()),
                default=DEFAULT_DATA_BACKEND,
                help="The backend used to read/write data")
        result = f
        #result = backend_args(result)
        result = backend(result)
        return result

    return wrapper

def layer_argument(name, allowed_types=None, default_type=None, required=True):
    def wrapper(f):
        ltypes = allowed_types
        dltype = default_type

        if ltypes is None:
            ltypes = get_layer_types()
        else:
            for t in ltypes:
                if t not in get_layer_types():
                    raise exceptions.IncorrectArgumentDefinition(str(f), name, argtype="layer",
                            reason="'{}' is not an allowed layer type".format(t))

        if dltype is None:
            dltype = DEFAULT_LAYER_TYPE

        if dltype not in ltypes:
            dltype = ltypes[0]

        backend = backend_argument(name)
        result = backend(f)

        layer_args = corgie_option('--{}_layer_args'.format(name),
                type=str,
                default='{}',
                help="JSON string describing additional layers args")
        result = layer_args(result)

        layer_type = corgie_option('--{}_layer_type'.format(name),
                type=click.Choice(ltypes),
                default=dltype)
        result = layer_type(result)

        path = corgie_option('--{}_path'.format(name), nargs=1,
                type=str, required=required)
        result = path(result)

        return result

    return wrapper

corgie_layer_argument = layer_argument

def create_data_backend_from_args(name=None, args_dict={},
        reference=None):
    prefix = ''
    if name is not None:
        prefix = '{}_'.format(name)

    if '{}backend'.format(prefix) not in args_dict:
        if reference is not None:
            backend = reference
            corgie_logger.debug("Using reference backend {}".format(
                backend
                ))
        else:
            backend_type = str_to_backend(DEFAULT_DATA_BACKEND)
            backend = backend_type()#**backend_args)
            corgie_logger.debug("Using default backend {}".format(
                backend
                ))
    else:
        backend_name = args_dict['{}backend'.format(prefix)]
        #backend_args = json.loads(args_dict['{}_backend_args'.format(name)])
        backend_type = str_to_backend(backend_name)
        backend = backend_type()#**backend_args)
        corgie_logger.debug("Using specified backend {}".format(
            backend
            ))

    return backend

def create_layer_from_args(name=None, args_dict={}, reference=None, must_have_layer_args=[],
        **kwargs):
    prefix = ''
    if name is not None:
        prefix = '{}_'.format(name)

    path_key = '{}path'.format(prefix)
    if path_key not in args_dict or \
            args_dict[path_key] is None:
        corgie_logger.debug("Path not specified for layer "
                "with name {}".format(name))
        return None
    layer_path = args_dict['{}path'.format(prefix)]

    type_key = '{}layer_type'.format(prefix)
    if type_key not in args_dict:
        corgie_logger.debug("Layer type not specified")
        if reference is not None:
            layer_type = str(reference)
            corgie_logger.debug("Using referency type {}".format(
                layer_type
                ))
        else:
            raise exceptions.ArgumentError('layer_type '
                    'for layer {}'.format(name), 'not given')
    else:
        layer_type = args_dict[type_key]
        corgie_logger.debug("Using specified type {}".format(
            layer_type
            ))

    args_key = '{}layer_args'.format(prefix)
    layer_args = {}
    if args_key in args_dict:
        layer_args = json.loads(args_dict[args_key])
    for arg in must_have_layer_args:
        if arg not in layer_args:
            raise exceptions.ArgumentError(
                '{} layer_args'.format(name),
                '"{}" arg is required by this command'.format(arg))

    backend_reference = None
    if reference is not None:
        backend_reference = type(reference)
    backend = create_data_backend_from_args(name,
            args_dict, reference=backend_reference)

    layer = backend.create_layer(layer_path, layer_type=layer_type,
        reference=reference, layer_args=layer_args, **kwargs)

    return layer
