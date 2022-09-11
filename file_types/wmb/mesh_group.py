from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from file_types.shared import Vec3
from utils.file_utils import IBuffer


@dataclass(slots=True)
class WMBGroupedMesh:
    vertex_index_buffer_id: int
    mesh_group_id: int
    material_id: int
    col_tree_node_id: int
    mesh_group_info_material_pair: int
    unknown_world_data_id: int

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        return cls(*buffer.read_fmt('3IiIi'))


@dataclass(slots=True)
class WMBMeshGroupInfo:
    name: str
    lod_level: int
    mesh_start: int
    mesh_grouped_array: List[WMBGroupedMesh]

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        name_offset, lod_level, mesh_start, grouped_offset, mesh_count = buffer.read_fmt('Ii3I')

        with buffer.read_from_offset(grouped_offset):
            grouped_meshes = [WMBGroupedMesh.from_buffer(buffer) for _ in range(mesh_count)]
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
        return cls(name, lod_level, mesh_start, grouped_meshes)


@dataclass(slots=True)
class WMBMeshGroup:
    name: str
    bbox: npt.NDArray[Vec3]
    material_indices: npt.NDArray[np.uint16]
    bone_indices: npt.NDArray[np.uint16]

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        name_offset = buffer.read_uint32()
        bbox = np.fromfile(buffer, Vec3, 2)
        material_index_offset, material_index_count = buffer.read_fmt('2I')
        bone_indices_offset, bone_indices_count = buffer.read_fmt('2I')
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
        with buffer.read_from_offset(material_index_offset):
            material_indices = np.fromfile(buffer, np.uint16, material_index_count)
        with buffer.read_from_offset(bone_indices_offset):
            bone_indices = np.fromfile(buffer, np.uint16, bone_indices_count)

        return cls(name, bbox, material_indices, bone_indices)
