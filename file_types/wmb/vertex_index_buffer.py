from dataclasses import dataclass
from enum import IntFlag, IntEnum
from typing import Optional, Union

import numpy as np
from numpy import typing as npt

from ...utils.file_utils import IBuffer


class WMBFlags(IntFlag):
    INT_FACES = 0x8


class WMBVertexVariant(IntEnum):
    NORMAL = 0
    UV2_ENORMAL = 1
    UV2_COLOR_ENORMAL = 4
    UV2_COLOR_ENORMAL_EUV3 = 5
    SKINNING_EUV2_EXNORMAL = 7
    SKINNING_EUV2_ECOLOR_ENORMAL = 10
    SKINNING_EUV2_ECOLOR_ENORMAL_EUV3 = 11
    UV2_COLOR_ENORMAL_EUV3_EUV4_EUV5 = 12
    UV2_COLOR_ENORMAL_EUV3_EUV4 = 14


@dataclass(slots=True)
class WMBVertexIndexBuffer:
    vertices: npt.NDArray
    vertices_extra: Optional[npt.NDArray]
    indices: Union[npt.NDArray[np.uint16], npt.NDArray[np.uint32]]

    @classmethod
    def from_buffer(cls, buffer: IBuffer, int32_indices: bool):
        vertex_data_offset, vertex_extra_data_offset = buffer.read_fmt('2I')
        unk08, unk0c = buffer.read_fmt('2I')
        vertex_stride, vertex_extra_stride = buffer.read_fmt('2I')
        unk18, unk1c = buffer.read_fmt('2I')
        vertex_count, vertex_variant = buffer.read_fmt('2I')
        vertex_variant = WMBVertexVariant(vertex_variant)
        indices_data_offset, indices_count = buffer.read_fmt('2I')

        assert unk08 == 0
        assert unk0c == 0
        assert unk18 == 0
        assert unk1c == 0

        vertex_elements = [
            ('pos', np.float32, 3),
            ('unk', np.int8, 4),
            ('uv', np.float16, 2),
        ]
        if vertex_variant == WMBVertexVariant.NORMAL:
            vertex_elements.append(('normal', np.float16, 4))
        if vertex_variant in (WMBVertexVariant.UV2_ENORMAL,
                              WMBVertexVariant.UV2_COLOR_ENORMAL,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4_EUV5,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4):
            vertex_elements.append(('uv2', np.float16, 2))
        if vertex_variant in (WMBVertexVariant.SKINNING_EUV2_EXNORMAL,
                              WMBVertexVariant.SKINNING_EUV2_ECOLOR_ENORMAL,
                              WMBVertexVariant.SKINNING_EUV2_ECOLOR_ENORMAL_EUV3):
            vertex_elements.append(('bone_indices', np.uint8, 4))
            vertex_elements.append(('bone_weights', np.uint8, 4))
        if vertex_variant in (WMBVertexVariant.UV2_COLOR_ENORMAL,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4_EUV5,
                              WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4):
            vertex_elements.append(('color', np.uint8, 4))

        vertex_dtype = np.dtype(vertex_elements)
        assert vertex_dtype.itemsize == vertex_stride
        with buffer.read_from_offset(vertex_data_offset):
            vertices = np.frombuffer(buffer.read(vertex_count*vertex_dtype.itemsize), vertex_dtype, vertex_count)

        if vertex_variant != 0:
            evertex_elements = []
            if vertex_variant in (WMBVertexVariant.UV2_ENORMAL, WMBVertexVariant.UV2_COLOR_ENORMAL):
                evertex_elements.append(('normal', np.float16, 4))
            elif vertex_variant == WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3:
                evertex_elements.append(('normal', np.float16, 4))
                evertex_elements.append(('uv3', np.float16, 2))
            elif vertex_variant == WMBVertexVariant.SKINNING_EUV2_EXNORMAL:
                evertex_elements.append(('uv2', np.float16, 2))
                evertex_elements.append(('normal', np.float16, 4))
            elif vertex_variant == WMBVertexVariant.SKINNING_EUV2_ECOLOR_ENORMAL:
                evertex_elements.append(('uv2', np.float16, 2))
                evertex_elements.append(('color', np.uint8, 4))
                evertex_elements.append(('normal', np.float16, 4))
            elif vertex_variant == WMBVertexVariant.SKINNING_EUV2_ECOLOR_ENORMAL_EUV3:
                evertex_elements.append(('uv2', np.float16, 2))
                evertex_elements.append(('color', np.uint8, 4))
                evertex_elements.append(('normal', np.float16, 4))
                evertex_elements.append(('uv3', np.float16, 2))
            elif vertex_variant == WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4_EUV5:
                evertex_elements.append(('uv2', np.float16, 2))
                evertex_elements.append(('uv3', np.float16, 2))
                evertex_elements.append(('uv4', np.float16, 2))
                evertex_elements.append(('uv5', np.float16, 2))
            elif vertex_variant == WMBVertexVariant.UV2_COLOR_ENORMAL_EUV3_EUV4:
                evertex_elements.append(('normal', np.float16, 4))
                evertex_elements.append(('uv3', np.float16, 2))
                evertex_elements.append(('uv4', np.float16, 2))

            evertex_dtype = np.dtype(evertex_elements)
            assert evertex_dtype.itemsize == vertex_extra_stride
            with buffer.read_from_offset(vertex_extra_data_offset):
                extra_vertices = np.frombuffer(buffer.read(vertex_count*evertex_dtype.itemsize), evertex_dtype, vertex_count)
        else:
            extra_vertices = None
        with buffer.read_from_offset(indices_data_offset):
            index_dtype = np.uint32 if int32_indices else np.uint16
            indices = np.frombuffer(buffer.read(indices_count*np.dtype(index_dtype).itemsize), index_dtype, indices_count)
        return cls(vertices, extra_vertices, indices)
