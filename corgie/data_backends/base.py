import torch

from corgie import exceptions

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, str_to_layer_type

#TODO: DEFNITELY need a full blown cache.

STR_TO_BACKEND_DICT = {}


def str_to_backend(s):
    global STR_TO_BACKEND_DICT
    return STR_TO_BACKEND_DICT[s]


def get_data_backends():
    return list(STR_TO_BACKEND_DICT.keys())


def register_backend(name):
    def register_backend_fn(cls):
        global STR_TO_BACKEND_DICT
        STR_TO_BACKEND_DICT[name] = cls
        return cls
    return register_backend_fn


class DataBackendBase:
    layer_constr_dict = {n: None for n in get_layer_types()}
    default_device = None

    def __init__(self, *kargs, device=None, **kwargs):
        if device is not None:
            self.device = device
        else:
            self.device = self.default_device
        super().__init__(*kargs, **kwargs)

    def create_layer(self, path, layer_type, *kargs, reference=None, **kwargs):
        if layer_type not in self.layer_constr_dict:
            raise Exception("Layer type {} is not \
                    defined".format(layer_type))
        if self.layer_constr_dict[layer_type] is None:
            raise Exception("Layer type {} is not \
                    implemented for {} backend".format(layer_type, type(self)))

        corgie_logger.debug("Creating layer '{}' on device '{}' with reference '{}'...".format(
                path, self.device, reference
                ))
        layer = self.layer_constr_dict[layer_type](path, device=self.device,
                reference=reference,
                backend=self,
                *kargs,
                **kwargs)
        corgie_logger.debug("Done")
        return layer

    @classmethod
    def register_layer_type_backend(cls, layer_type_name):
        # This is a decorator for including
        assert layer_type_name in cls.layer_constr_dict

        def register_fn(layer):
            cls.layer_constr_dict[layer_type_name] = layer
            return layer

        return register_fn


class BaseLayerBackend:
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def get_sublayer(obj, *kargs, **kwargs):
        raise Exception("layer type backend must implement "
                "'get_sublayer' function")

        #$def get_data_type(self):
    #    raise Exception("layer type backend must implement "
    #            "'get_data_type' function")

    def read_backend(self, *kargs, **kwargs):
        raise Exception("layer type backend must implement "
                "'read_backend' function")

    def write_backend(self, *kargs, **kwargs):
        raise Exception("layer type backend must implement"
                "'write_backend' function")

