from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from utils.file_utils import FileBuffer, IBuffer


@dataclass(slots=True)
class TextureInfo:
    offset: int
    size: int
    unk1: int
    unk2: int
    id: int


@dataclass(slots=True)
class WTA:
    texture_info: Dict[int, TextureInfo]

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        self = cls({})
        magic = buffer.read_ascii_string(4)
        assert magic == 'WTB', f'Invalid WTB magic, got {magic}'
        (version, texture_count, textures_offsets_offset, textures_size_offset,
         textures_unk1_offset, textures_id_offset, textures_unk2_offset) = buffer.read_fmt('7I')
        assert version == 3
        buffer.seek(textures_offsets_offset)
        texture_offsets = buffer.read_fmt(f'{texture_count}I')
        buffer.seek(textures_size_offset)
        texture_sizes = buffer.read_fmt(f'{texture_count}I')
        buffer.seek(textures_unk1_offset)
        texture_unk1 = buffer.read_fmt(f'{texture_count}I')
        buffer.seek(textures_unk2_offset)
        texture_unk2 = buffer.read_fmt(f'{texture_count}I')
        buffer.seek(textures_id_offset)
        texture_ids = buffer.read_fmt(f'{texture_count}I')
        for offset, size, unk1, unk2, tid in zip(texture_offsets, texture_sizes,
                                                 texture_unk1, texture_unk2,
                                                 texture_ids):
            self.texture_info[tid] = TextureInfo(offset, size, unk1, unk2, tid)
        return self


def wta_from_path(path: Path):
    return wta_from_buffer(FileBuffer(path, 'rb'))


def wta_from_buffer(buffer: IBuffer):
    return WTA.from_buffer(buffer)
