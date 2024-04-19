from decimal import Decimal
from math import ceil, log10
from timeit import default_timer

import ipdb

_active_timers: list["Timer"] = []
_all_time: dict[str, Decimal] = {}
_all_cnt: dict[str, int] = {}
_enabled: bool = True

_prefix_stack: list[str] = []


def _prefix():
    if len(_prefix_stack) == 0:
        return ""
    return ".".join(_prefix_stack) + "."


class Timer:
    def __init__(self, name: str, print_results=True, parent: "Timer" = None):
        if len(_prefix_stack) > 0:
            name = _prefix() + name

        self.elapsed = Decimal()
        self._name = name
        self._print_results = print_results
        self._start_time = None
        self._children = {}
        self._parent = parent
        self._count = 0
        self.started = False

        # Ignore the first 5 times.
        self.ignore_k = 10

    @staticmethod
    def push_prefix(prefix: str):
        _prefix_stack.append(prefix)

    @staticmethod
    def pop_prefix():
        _prefix_stack.pop()

    @staticmethod
    def disable():
        global _enabled
        _enabled = False

    @staticmethod
    def enable():
        global _enabled
        _enabled = True

    @staticmethod
    def get_active() -> "Timer":
        if len(_active_timers) > 0:
            return _active_timers[-1]

        return Timer("")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if self._print_results:
            self.print_results()

    def child(self, name: str):
        try:
            result = self._children[_prefix() + name]
            result._parent = self
            return result
        except KeyError:
            result = Timer(name, print_results=False, parent=self)
            self._children[result._name] = result
            return result

    def start(self):
        assert not self.started
        self._count += 1
        self._start_time = self._get_time()
        # If we aren't a child of the active timer, then add us to the active timer's children.
        global _active_timers
        if len(_active_timers) > 0:
            if _active_timers[-1] is not self:
                _active_timers[-1]._children[self._name] = self
            self._parent = _active_timers[-1]

        _active_timers.append(self)
        self.started = True
        return self

    def stop(self):
        assert self.started

        if _enabled:
            # Only start counting if we are enabled.

            diff = self._get_time() - self._start_time
            self.elapsed += diff
            _all_cnt[self._name] = _all_cnt.get(self._name, 0) + 1

            # Only start counting after the first ignore_k times.
            if _all_cnt[self._name] > self.ignore_k:
                _all_time[self._name] = _all_time.get(self._name, 0) + diff

        # If we are the active timer, then set the active timer to our parent.
        global _active_timers
        if _active_timers[-1] is self:
            _active_timers.pop()
        self._parent = None
        self.started = False
        return self

    @property
    def elapsed_mean(self):
        if _all_cnt[self._name] > self.ignore_k:
            return _all_time[self._name] / (_all_cnt[self._name] - self.ignore_k)
        else:
            return self.elapsed

    def print_results(self):
        # print(self._format_results())
        pass

    def _format_results(self, indent="  "):
        children = self._children.values()
        elapsed = self.elapsed_mean or sum(c.elapsed_mean for c in children)

        result = "%s: %.5fs" % (self._name, elapsed)
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(ceil(log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c.elapsed_mean, reverse=True):
            lines = child._format_results(indent).split("\n")

            child_percent = child.elapsed_mean / elapsed * 100

            if child_percent > 100:
                print("??")
                ipdb.set_trace()

            lines[0] += " (%d%%)" % child_percent
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = ("%dx " % child._count).rjust(count_digits + 2) + lines[0]
            for line in lines:
                result += "\n" + indent + line
        return result

    def _get_time(self):
        return Decimal(default_timer())
