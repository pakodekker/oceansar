"""Measured memory budgets for isolated simulator stages and notebook calls."""
import contextlib
import csv
import ctypes
from functools import lru_cache, wraps
import importlib
import inspect
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback


GIB = 1024**3
_MONITORED_WORKER = False


class SimulationMemoryError(MemoryError):
    """A monitored simulator worker reached its configured memory budget."""


@lru_cache(maxsize=1)
def _mac_usage_api():
    class Usage(ctypes.Structure):
        _fields_ = [('uuid', ctypes.c_uint8*16)]+[(name, ctypes.c_uint64) for name in (
            'user_time','system_time','pkg_wakeups','interrupt_wakeups','pageins',
            'wired','resident','footprint','start_time','exit_time')]
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    return library, Usage


def process_memory(pid):
    """Read resident/accounted bytes for one Python process without its arrays.

    On macOS, account max(RSS, physical footprint); footprint includes charged
    compressed memory that RSS alone misses. Linux adds VmSwap to RSS, and
    Windows accounts max(RSS, private committed bytes). Return a small dict.
    Measurement failure propagates, so a guarded run cannot silently continue
    without its monitor. The simulator uses threads, not extra worker processes.
    """
    if sys.platform == 'darwin':
        library, Usage = _mac_usage_api()
        usage = Usage()
        if library.proc_pid_rusage(int(pid),0,ctypes.byref(usage)):
            code = ctypes.get_errno()
            raise OSError(code,os.strerror(code))
        return dict(rss_bytes=int(usage.resident),
                    accounted_bytes=int(max(usage.resident,usage.footprint)))
    import psutil
    memory = psutil.Process(pid).memory_info()
    accounted = max(memory.rss,getattr(memory,'private',memory.rss))
    if sys.platform.startswith('linux'):
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmSwap:'):
                accounted += int(line.split()[1])*1024
                break
    return dict(rss_bytes=int(memory.rss),accounted_bytes=int(accounted))


def configured_memory_limit(cfg_file):
    """Return sim.memory_limit_gib, or None when absent/None/zero (disabled).

    cfg_file is an OceanSAR configuration path, optionally None. A positive
    finite number sets a budget in GiB for the calling Python process plus
    its isolated simulator worker. Negative, boolean and nonfinite values
    raise ValueError. VS Code and unrelated kernels/apps are outside this scope.
    """
    if cfg_file is None:
        return None
    from oceansar import ocs_io
    cfg = ocs_io.ConfigFile(str(cfg_file))
    value = getattr(getattr(cfg,'sim',None),'memory_limit_gib',None)
    if value is None or str(value).lower() == 'none':
        return None
    if isinstance(value,bool):
        raise ValueError('sim.memory_limit_gib must be a finite positive number, or 0/None')
    value = float(value)
    if not 0 <= value < float('inf'):
        raise ValueError('sim.memory_limit_gib must be a finite positive number, or 0/None')
    return value or None


class _Stream:
    def __init__(self, connection, channel):
        self.connection,self.channel = connection,channel
    def write(self, text):
        if text:
            self.connection.send((self.channel,str(text)))
        return len(text)
    def flush(self):
        pass
    def isatty(self):
        return False


def _worker(connection, module, name, args, kwargs):
    global _MONITORED_WORKER
    _MONITORED_WORKER = True
    os.environ['MPLBACKEND'] = 'Agg'
    with contextlib.redirect_stdout(_Stream(connection,'stdout')), \
         contextlib.redirect_stderr(_Stream(connection,'stderr')):
        try:
            function = getattr(importlib.import_module(module),name)
            if 'matplotlib.pyplot' in sys.modules:
                import matplotlib
                matplotlib.use('Agg',force=True)
                from matplotlib import pyplot as plt
                def show(*args, **kwargs):
                    for number in plt.get_fignums():
                        buffer = io.BytesIO()
                        plt.figure(number).savefig(buffer,format='png')
                        connection.send(('figure',buffer.getvalue()))
                plt.show = show
            result = function(*args,**kwargs)
            connection.send(('result',result))
        except BaseException as error:
            try:
                connection.send(('error',(error,traceback.format_exc())))
            except Exception:
                connection.send(('error',(RuntimeError(str(error)),traceback.format_exc())))
        finally:
            connection.close()


def _stop(worker):
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=2)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=2)


def run_with_memory_limit(function, *args, limit_gib, report_path=None, **kwargs):
    """Execute an importable function in a spawned, externally monitored worker.

    function is a module-level callable; args/kwargs and its result must be
    pickleable. limit_gib is a positive GiB budget for caller plus worker.
    Include caller memory so a notebook retaining large arrays cannot quietly
    consume the worker's entire budget too. Return the function's result and
    relay stdout/stderr and explicit notebook figures. Nested guarded calls
    execute inside the already monitored worker, avoiding duplicate processes.

    Read memory every 0.05 s, stop at 95% of the budget, and raise
    SimulationMemoryError in the caller; the notebook kernel survives. This
    is a sampled safety guard, not a byte-exact OS allocation cap: a rapid
    native allocation can overshoot between polls. Existing/partial products
    are retained for inspection. Optional report_path writes measured peaks
    and a once-per-second CSV, including failures and user interruption.
    """
    if _MONITORED_WORKER:
        return function(*args,**kwargs)
    limit_gib = float(limit_gib)
    if not 0 < limit_gib < float('inf'):
        raise ValueError('Memory limit must be finite and positive')
    budget = int(limit_gib*GIB)
    threshold = int(.95*budget)
    caller = process_memory(os.getpid())
    if caller['accounted_bytes'] >= threshold:
        raise SimulationMemoryError(f'Calling Python process already uses '
            f'{caller["accounted_bytes"]/GIB:.2f} GiB; configured limit is {limit_gib:g} GiB.')
    module = function.__module__
    if module == '__main__':
        spec = inspect.unwrap(function).__globals__.get('__spec__')
        if spec is not None:
            module = spec.name
    context = multiprocessing.get_context('spawn')
    receive,send = context.Pipe(duplex=False)
    worker = context.Process(target=_worker,args=(send,module,function.__name__,args,kwargs))
    report = dict(function=f'{module}.{function.__name__}',limit_gib=limit_gib,
        stop_fraction=.95,poll_interval_seconds=.05,state='starting',
        peak_worker_accounted_gib=0.,peak_worker_rss_gib=0.,peak_combined_gib=caller['accounted_bytes']/GIB,
        scope='calling Python process plus isolated worker; excludes editor and unrelated apps')
    report_path = None if report_path is None else Path(report_path)
    csv_handle = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        csv_handle = report_path.with_suffix('.csv').open('w',newline='')
        writer = csv.writer(csv_handle)
        writer.writerow(('elapsed_s','worker_rss_gib','worker_accounted_gib','caller_accounted_gib','combined_gib'))
    started = last_sample = time.monotonic()
    result = None
    completed = False
    try:
        worker.start()
        send.close()
        report.update(state='running',worker_pid=worker.pid)
        next_check = 0.
        while True:
            now = time.monotonic()
            if now >= next_check and worker.is_alive():
                try:
                    memory = process_memory(worker.pid)
                except (ProcessLookupError,FileNotFoundError):
                    memory = dict(rss_bytes=0,accounted_bytes=0)
                caller = process_memory(os.getpid())
                combined = memory['accounted_bytes']+caller['accounted_bytes']
                report['peak_worker_rss_gib'] = max(report['peak_worker_rss_gib'],memory['rss_bytes']/GIB)
                report['peak_worker_accounted_gib'] = max(report['peak_worker_accounted_gib'],memory['accounted_bytes']/GIB)
                report['peak_combined_gib'] = max(report['peak_combined_gib'],combined/GIB)
                if csv_handle is not None and (now-last_sample>=1 or combined>=threshold):
                    writer.writerow((now-started,memory['rss_bytes']/GIB,memory['accounted_bytes']/GIB,
                                     caller['accounted_bytes']/GIB,combined/GIB))
                    csv_handle.flush();last_sample=now
                if combined >= threshold:
                    report['state'] = 'memory_limit_exceeded'
                    _stop(worker)
                    raise SimulationMemoryError(f'{function.__name__} stopped at measured '
                        f'{combined/GIB:.2f} GiB (caller + worker); configured limit '
                        f'{limit_gib:g} GiB. Partial products retained; see {report_path}.')
                next_check = now+.05
            if receive.poll(.01):
                try:
                    kind,value = receive.recv()
                except EOFError:
                    break
                if kind in ('stdout','stderr'):
                    getattr(sys,kind).write(value)
                    getattr(sys,kind).flush()
                elif kind == 'figure':
                    try:
                        from IPython import get_ipython
                    except ImportError:
                        get_ipython = lambda: None
                    if get_ipython() is not None:
                        from IPython.display import Image,display
                        display(Image(data=value))
                elif kind == 'error':
                    error,trace = value
                    if hasattr(error,'add_note'):
                        error.add_note('Isolated simulator traceback:\n'+trace)
                    raise error
                elif kind == 'result':
                    result,completed = value,True
            elif not worker.is_alive():
                break
        worker.join(timeout=2)
        if not completed or worker.exitcode:
            raise RuntimeError(f'{function.__name__} worker exited with code {worker.exitcode} before completing')
        report['state'] = 'complete'
        return result
    except BaseException as error:
        if report['state'] != 'memory_limit_exceeded':
            report['state'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        _stop(worker)
        if worker.pid is not None:
            worker.close()
        receive.close();send.close()
        if csv_handle is not None:
            csv_handle.close()
        report['elapsed_seconds'] = time.monotonic()-started
        if report_path is not None:
            report_path.write_text(json.dumps(report,indent=2)+'\n')


def memory_limited(function):
    """Apply sim.memory_limit_gib to a public function accepting cfg_file.

    Missing/zero/None preserves in-process execution and existing mocking or
    profiling workflows. A positive setting isolates the function, measures
    actual memory, returns its result, and frees worker arrays after the call.
    Write memory_<function>.json/csv beside its output, or in sim.path.
    """
    signature = inspect.signature(function)
    @wraps(function)
    def guarded(*args,**kwargs):
        if _MONITORED_WORKER:
            return function(*args,**kwargs)
        bound = signature.bind_partial(*args,**kwargs)
        cfg_file = bound.arguments.get('cfg_file')
        limit = configured_memory_limit(cfg_file)
        if limit is None:
            return function(*args,**kwargs)
        from oceansar import ocs_io
        cfg = ocs_io.ConfigFile(str(cfg_file))
        output = bound.arguments.get('output_file',bound.arguments.get('output_raw'))
        folder = Path(output).parent if output is not None else Path(cfg.sim.path)
        return run_with_memory_limit(guarded,*args,limit_gib=limit,
            report_path=folder/f'memory_{function.__name__}.json',**kwargs)
    return guarded
