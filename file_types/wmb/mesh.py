from dataclasses import dataclass

from utils.file_utils import IBuffer


@dataclass(slots=True)
class WMBMesh:
    vertex_index_buffer_id: int
    bone_set_id: int
    vertex_start: int
    indices_offset: int
    vertex_count: int
    indices_count: int
    triangle_count: int

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        self = cls(*buffer.read_fmt('7I'))
        assert self.indices_count // 3 == self.triangle_count
        return self
