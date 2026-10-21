"""Opt-in interpreter switch interval for the API adapter process only.

Importing this module has no scheduling side effects. Configure once when the
actual frontend adapter is constructed, after the engine workers have started.
A subsequently forked child restores the original interval before it can run
worker code. Spawned children retain Python's ordinary interval unless they
construct their own API adapter.
"""

import os
import sys
import threading

_LOCK = threading.Lock()
_STATE = None
_FORK_REGISTERED = False
_ENV = 'EASYMAGPIE_API_SWITCH_INTERVAL_MS'


def _after_fork_child():
    global _LOCK, _STATE
    _LOCK = threading.Lock()
    state, _STATE = _STATE, None
    if state is not None and state['pid'] != os.getpid():
        sys.setswitchinterval(state['original_seconds'])


def configure_api_switch(ctx):
    """Set 1ms once per API process; no change for the default-off setting."""
    global _STATE, _FORK_REGISTERED
    option = os.environ.get(_ENV, '0')
    if option not in ('0', '1'):
        raise ValueError(f'{_ENV} must be 0 (off) or 1 (one millisecond)')
    if option == '0':
        return api_switch_status()
    if getattr(ctx, 'engine_client', None) is None:
        raise RuntimeError('API switch interval requires the serving adapter engine context')
    with _LOCK:
        if _STATE is not None:
            if _STATE['pid'] != os.getpid():
                raise RuntimeError('API scheduling state inherited without fork restoration')
            return api_switch_status()
        before = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        _STATE = {
            'pid': os.getpid(),
            'original_seconds': before,
            'requested_seconds': 0.001,
            'configured_seconds': sys.getswitchinterval(),
            'configuration_count': 1,
            'scope': 'API adapter interpreter process',
        }
        if hasattr(os, 'register_at_fork') and not _FORK_REGISTERED:
            os.register_at_fork(after_in_child=_after_fork_child)
            _FORK_REGISTERED = True
        return api_switch_status()


def api_switch_status():
    return {
        'enabled': _STATE is not None,
        'current_pid': os.getpid(),
        'current_seconds': sys.getswitchinterval(),
        'configuration': dict(_STATE) if _STATE else None,
    }
