from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np

from ..shared import Vec3
from ...utils.file_utils import IBuffer, FileBuffer


@dataclass
class ModelRef:
    dir: str
    id: int

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        return cls(buffer.read_ascii_string(2), buffer.read_uint16())

    @property
    def name(self):
        return f"{self.dir}{self.id:04x}"

    @property
    def path(self):
        return Path(f"{self.dir}/{self.dir}{self.id:04x}")

    def __str__(self):
        return f"Model(\"{self.name}\",\"{self.path}\")"

    def __repr__(self):
        return f"Model(\"{self.name}\",\"{self.path}\")"


@dataclass
class Instance:
    pos: Vec3
    rot: Vec3
    scl: Vec3

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        return cls(np.frombuffer(buffer.read(12), Vec3)[0],
                   np.frombuffer(buffer.read(12), Vec3)[0],
                   np.frombuffer(buffer.read(12), Vec3)[0])


@dataclass
class Asset:
    name: str
    pos: Vec3
    rot: Vec3
    scl: Vec3

    index: int
    instance_count: int
    instances: List[Instance] = field(default_factory=list)

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        name, pos, rot, scl = (buffer.read_ascii_string(32), np.frombuffer(buffer.read(12), Vec3)[0],
                               np.frombuffer(buffer.read(12), Vec3)[0],
                               np.frombuffer(buffer.read(12), Vec3)[0],)
        uint_ = buffer.read_uint32()
        assert uint_ == 0
        index = buffer.read_uint32()
        assert not all([buffer.read_uint32() for _ in range(8)])
        instance_count = buffer.read_uint32()
        return cls(name, pos, rot, scl, index, instance_count)


class Layout:
    def __init__(self):
        self.models_refs: Dict[str, ModelRef] = {}
        self.assets: List[Asset] = []

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        assert buffer.read_ascii_string(4) == "LAY"
        version = round(buffer.read_float(), 3)
        assert version == 2.01
        (
            models_offset, model_count,
            assets_offset, asset_count,
            instances_offset, instance_count,
        ) = buffer.read_fmt('6I')
        buffer.seek(models_offset)
        models = [ModelRef.from_buffer(buffer) for _ in range(model_count)]
        buffer.seek(assets_offset)
        assets = [Asset.from_buffer(buffer) for _ in range(asset_count)]
        buffer.seek(instances_offset)
        instances = [Instance.from_buffer(buffer) for _ in range(instance_count)]
        for asset in assets:
            asset.instances.extend([instances.pop(0) for _ in range(asset.instance_count)])
        self = cls()
        self.models_refs = {model.name: model for model in models}
        self.assets = assets
        return self

    def find_model_ref_by_model_name(self, model_name):
        return self.models_refs.get(model_name, None)


def lay_from_buffer(buffer: IBuffer):
    return Layout.from_buffer(buffer)


def lay_from_path(path: Path):
    with FileBuffer(path, 'rb') as f:
        return lay_from_buffer(f)
