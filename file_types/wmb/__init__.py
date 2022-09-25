from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..shared import Vec3
from .bone import WMBBoneSet, WMBBone
from .material import WMBMaterial
from .mesh import WMBMesh
from .mesh_group import WMBMeshGroupInfo, WMBMeshGroup
from .vertex_index_buffer import WMBFlags, WMBVertexIndexBuffer
from ...utils.file_utils import IBuffer, FileBuffer


class WMB:

    def __init__(self):
        self.version = 0
        self.flags: WMBFlags = WMBFlags(0)
        self.bbox: Vec3 = np.zeros(2, Vec3)
        self.bones: List[WMBBone] = []
        self.materials: List[WMBMaterial] = []
        self.meshes: List[WMBMesh] = []
        self.mesh_group_infos: List[WMBMeshGroupInfo] = []
        self.mesh_groups: List[WMBMeshGroup] = []
        self.bone_map: List[int] = []
        self.bone_sets: List[WMBBoneSet] = []
        self.mesh_material_pairs: List[Tuple[int, int]] = []
        self.vertex_index_buffers: List[WMBVertexIndexBuffer] = []

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        self = cls()
        magic = buffer.read_ascii_string(4)
        assert magic == 'WMB3', f'Invalid WMB3 magic, got {magic}'
        self.version = buffer.read_uint32()
        zero = buffer.read_uint32()
        assert zero == 0, f'Should be zero, got {zero}'
        self.flags = WMBFlags(buffer.read_uint32())
        self.bbox = np.frombuffer(buffer.read(12 * 2), Vec3, 2)
        bone_offset, bone_count = buffer.read_fmt('2I')
        buffer.read_fmt('2I')
        vertex_index_buffers_offset, vertex_index_buffers_count = buffer.read_fmt('2I')
        meshes_offset, meshes_count = buffer.read_fmt('2I')
        mesh_groups_info_offset, mesh_groups_info_count = buffer.read_fmt('2I')
        collision_tree_offset, collision_tree_count = buffer.read_fmt('2I')
        bone_map_offset, bone_map_count = buffer.read_fmt('2I')
        bone_sets_offset, bone_sets_count = buffer.read_fmt('2I')
        materials_offset, materials_count = buffer.read_fmt('2I')
        mesh_groups_offset, mesh_groups_count = buffer.read_fmt('2I')
        mesh_materials_offset, mesh_materials_count = buffer.read_fmt('2I')
        unk_world_data_offset, unk_world_data_count = buffer.read_fmt('2I')

        buffer.seek(bone_offset)
        self.bones = [WMBBone.from_buffer(buffer) for _ in range(bone_count)]
        buffer.seek(materials_offset)
        self.materials = [WMBMaterial.from_buffer(buffer) for _ in range(materials_count)]
        buffer.seek(vertex_index_buffers_offset)
        self.vertex_index_buffers = [WMBVertexIndexBuffer.from_buffer(buffer, self.flags & WMBFlags.INT_FACES != 0) for
                                     _ in
                                     range(vertex_index_buffers_count)]
        buffer.seek(meshes_offset)
        self.meshes = [WMBMesh.from_buffer(buffer) for _ in range(meshes_count)]
        buffer.seek(mesh_groups_info_offset)
        self.mesh_group_infos = [WMBMeshGroupInfo.from_buffer(buffer) for _ in range(mesh_groups_info_count)]
        buffer.seek(mesh_groups_offset)
        self.mesh_groups = [WMBMeshGroup.from_buffer(buffer) for _ in range(mesh_groups_count)]
        buffer.seek(bone_map_offset)
        self.bone_map = [buffer.read_uint32() for _ in range(bone_map_count)]
        buffer.seek(bone_sets_offset)
        self.bone_sets = [WMBBoneSet.from_buffer(buffer) for _ in range(bone_sets_count)]
        buffer.seek(mesh_materials_offset)
        self.mesh_material_pairs = [buffer.read_fmt('2I') for _ in range(mesh_materials_count)]
        # TODO: ColTree stuff
        return self


def wmb_from_path(path: Path):
    return wmb_from_buffer(FileBuffer(path, 'rb'))


def wmb_from_buffer(buffer: IBuffer):
    return WMB.from_buffer(buffer)
