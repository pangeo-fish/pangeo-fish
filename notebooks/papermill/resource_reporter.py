"""IPython extension to report peak cpu and memory usage for every cell"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psutil


def _fmt_memory(memory: int) -> str:
    mb = memory / 2**20
    gb = memory / 2**30
    if gb > 2:
        memory_fmt = f"{gb:.0f}GiB"
    elif gb > 1:
        memory_fmt = f"{gb:.1f}GiB"
    else:
        memory_fmt = f"{mb:.0f}MiB"
    return memory_fmt


class ResourceCollector:
    """Report peak resources over time

    Reports resources for the whole container, not just this process.
    So it will capture local dask, etc.

    Reports _peak_ memory usage and average CPU usage for a cell
    """

    def __init__(self, sample_rate=0.1):
        self.sample_rate = 0.1

        self._pool = ThreadPoolExecutor(1)
        self._event = self._future = None
        self._cpu_count = psutil.cpu_count()

    def collect_one(self):
        """Collect one sample of memory"""
        # lots of choices here
        # virtual_memory().used is very high
        # virtual_memory().active is also too high
        # memory_info().rss high, but lower than above
        # memory_full_info().uss under counts
        # memory_full_info().pss seems to match `kubectl top pod` the best
        return sum(p.memory_full_info().pss for p in psutil.process_iter())

    def collect(self, event):
        """Constantly collect memory and cpu usage (in the background)"""
        self.collect_one()
        peak_mem = self.collect_one()
        psutil.cpu_percent()  # start cpu percent counter
        while not event.is_set():
            time.sleep(self.sample_rate)
            try:
                mem = self.collect_one()
            except Exception as e:
                # suppress errors collecting memory
                print("Error collecting memory: {e}")
                continue
            if mem > peak_mem:
                # track large allocations as they happen
                if mem > peak_mem + (1 * 2**30):
                    print(f"Memory increased {_fmt_memory(peak_mem)} -> {_fmt_memory(mem)}")
                peak_mem = mem
        avg_cpu = 0.01 * psutil.cpu_percent() * self._cpu_count
        return avg_cpu, peak_mem

    def start(self, info=None):
        """Start collecting in a background thread"""
        self._start_time = time.perf_counter()
        assert self._event is None
        self._event = Event()
        self._future = self._pool.submit(self.collect, self._event)
        self._start = time.perf_counter()

    def finish(self):
        """Finish collecting and return result"""
        self._duration = time.perf_counter() - self._start_time
        self._event.set()
        f = self._future
        self._start = None
        self._event = None
        self._future = None
        return f.result()

    def finish_and_report(self, result=None):
        """Finish collecting and report results"""
        if self._event is None:
            # first hook run
            return
        try:
            cpu, memory = self.finish()
        except Exception as e:
            print(f"Error collecting resources: {e}")
            return
        memory_fmt = _fmt_memory(memory)
        print(f"Usage: cpu={cpu:.1f}, peak mem={memory_fmt}, duration={self._duration:.0f}s")

    # let it be used as a context manager
    def __enter__(self):
        self.start()

    def __exit__(self, *exc_info):
        self.finish_and_report()


def load_ipython_extension(ip):
    collector = ResourceCollector()

    ip.events.register("pre_run_cell", collector.start)
    ip.events.register("post_run_cell", collector.finish_and_report)
