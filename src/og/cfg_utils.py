import functools as ft

import cattrs
import ipdb
from attrs import astuple, define
from cattrs.strategies import configure_tagged_union, include_subclasses


@define
class Cfg:
    @staticmethod
    def get_converter():
        def structure_float_or_sched(val, cl):
            if isinstance(val, float):
                return val

            return converter.structure(val, Schedule)

        from og.schedules import Schedule

        converter = cattrs.Converter()
        union_strategy = ft.partial(configure_tagged_union)

        converter.register_structure_hook(float | Schedule, structure_float_or_sched)

        include_subclasses(Schedule, converter, union_strategy=union_strategy)
        include_subclasses(Cfg, converter, union_strategy=union_strategy)

        return converter

    @classmethod
    def fromdict(cls, d, use_converter: bool = True):
        if use_converter:
            converter = Cfg.get_converter()
            return converter.structure(d, cls)

        return cattrs.structure(d, cls)

    def asdict(self):
        converter = Cfg.get_converter()
        d = converter.unstructure(self)
        return d

    def astuple(self):
        return astuple(self)
