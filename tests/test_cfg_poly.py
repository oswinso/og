import ipdb
from attrs import define

from og.cfg_utils import Cfg


@define
class CfgA(Cfg):
    a: int

    @property
    def b(self):
        return "a"


@define
class CfgB(Cfg):
    a: int

    @property
    def b(self):
        return "b"


@define
class CfgC(Cfg):
    a: int

    @property
    def b(self):
        return "c"


def main():
    cfg1 = CfgA(1)
    d1 = cfg1.asdict()
    print("d1:")
    print(d1)

    cfg1_ = Cfg.fromdict(d1)
    print("cfg1_: {}".format(cfg1_.b))
    print(cfg1_)

    # ---------------------------------

    cfg2 = CfgB(2)
    d2 = cfg2.asdict()
    print("d2:")
    print(d2)

    cfg2_ = Cfg.fromdict(d2)
    print("cfg2_: {}".format(cfg2_.b))
    print(cfg2_)

    # ---------------------------------

    cfg3 = CfgC(3)
    d3 = cfg3.asdict()
    print("d3:")
    print(d3)

    cfg3_ = Cfg.fromdict(d3)
    print("cfg3_: {}".format(cfg3_.b))
    print(cfg3_)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
