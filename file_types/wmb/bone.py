from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from ..shared import Vec3
from ...utils.file_utils import IBuffer


@dataclass(slots=True)
class WMBBoneSet:
    bone_ids: npt.NDArray[np.uint16]

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        offset, count = buffer.read_fmt('2I')
        with buffer.read_from_offset(offset):
            bone_ids = np.frombuffer(buffer.read(count * 2), np.uint16, count)
        return cls(bone_ids)


@dataclass(slots=True)
class WMBBone:
    id: int
    parent_id: int
    pos: Vec3
    rot: Vec3
    scl: Vec3

    wpos: Vec3
    wrot: Vec3
    wscl: Vec3

    wpos_tpose: Vec3

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        bone_id, parent_id = buffer.read_fmt('2h')
        pos = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        rot = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        scl = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        wpos = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        wrot = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        wscl = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        wpos_tpose = np.frombuffer(buffer.read(12), Vec3, 1)[0]
        return cls(bone_id, parent_id, pos, rot, scl, wpos, wrot, wscl, wpos_tpose)
