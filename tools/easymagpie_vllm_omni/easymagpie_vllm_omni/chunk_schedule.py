"""Immutable startup chunk schedule shared by the two CPU accumulators."""


def startup_schedule(values=(2, 6)):
    if not isinstance(values, (list, tuple)):
        raise ValueError('startup chunk frames must be a list or tuple')
    # Eight rows is the fixed accumulator capacity and terminal padding width.
    # Empty startup is valid: publish steady eight-frame chunks from the start.
    if len(values) > 4096 or any(type(value) is not int or not 1 <= value <= 8 for value in values):
        raise ValueError('startup chunk frames must contain at most 4096 integers in 1..8')
    return tuple(values)
