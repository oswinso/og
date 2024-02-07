import ipdb
from attrs import define

from og.cfg_utils import Cfg
from og.schedules import Constant, LinDecay, Schedule


@define
class TestCfg(Cfg):
    sched: Schedule


def main():
    cfg1 = TestCfg(Constant(1.234))
    d = cfg1.asdict()
    print(d)
    cfg1_ = TestCfg.fromdict(d)
    print(cfg1_)

    cfg2 = TestCfg(LinDecay(3.0, 4.0, 5, 6))
    d = cfg2.asdict()
    print(d)
    cfg2_ = TestCfg.fromdict(d)
    print(cfg2_)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
