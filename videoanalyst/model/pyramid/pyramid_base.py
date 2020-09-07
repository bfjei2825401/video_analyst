# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

TRACK_PYRAMIDS = Registry('TRACK_PYRAMIDS')
VOS_PYRAMIDS = Registry('VOS_PYRAMIDS')

TASK_PYRAMIDS = dict(
    track=TRACK_PYRAMIDS,
    vos=VOS_PYRAMIDS,
)
