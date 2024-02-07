import functools as ft

import cattrs
from attrs import astuple
from cattrs.strategies import configure_tagged_union, include_subclasses


class Cfg:
    @staticmethod
    def get_converter():
        from og.schedules import Schedule

        converter = cattrs.Converter()
        union_strategy = ft.partial(configure_tagged_union)
        include_subclasses(Schedule, converter, union_strategy=union_strategy)
        return converter

    @classmethod
    def fromdict(cls, d):
        return Cfg.get_converter().structure(d, cls)

    def asdict(self):
        d = Cfg.get_converter().unstructure(self)
        return d

    def astuple(self):
        return astuple(self)
