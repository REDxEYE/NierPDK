from dataclasses import dataclass
from typing import List

from utils.file_utils import IBuffer


@dataclass(slots=True)
class WMBMaterial:
    @dataclass(slots=True)
    class TextureParam:
        name: str
        value: float

        @classmethod
        def from_buffer(cls, buffer: IBuffer):
            name_offset = buffer.read_uint32()
            value = buffer.read_uint32()
            with buffer.read_from_offset(name_offset):
                name = buffer.read_ascii_string()
            return cls(name, value)

    @dataclass(slots=True)
    class UniformVariable:
        name: str
        ident: int

        @classmethod
        def from_buffer(cls, buffer: IBuffer):
            name_offset = buffer.read_uint32()
            ident = buffer.read_uint32()
            with buffer.read_from_offset(name_offset):
                name = buffer.read_ascii_string()
            return cls(name, ident)

    @dataclass(slots=True)
    class ParameterGroup:
        ident: int
        params: List[float]

        @classmethod
        def from_buffer(cls, buffer: IBuffer):
            ident, offset, count = buffer.read_fmt('3I')
            with buffer.read_from_offset(offset):
                params = buffer.read_fmt(f'{count}f')
            return cls(ident, params)

    unk: List[int]
    name: str
    effect_name: str
    technique_name: str
    unk2: int
    texture_params: List[TextureParam]
    parameter_groups: List[ParameterGroup]
    uniforms: List[UniformVariable]

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        unk = buffer.read_fmt('4h')
        name_offset, effect_offset, technique_offset = buffer.read_fmt('3I')
        unk2 = buffer.read_uint32()
        texture_offset, texture_count = buffer.read_fmt('2I')
        parameter_groups_offset, parameter_groups_count = buffer.read_fmt('2I')
        variables_offset, variables_count = buffer.read_fmt('2I')
        with buffer.save_current_offset():
            buffer.seek(texture_offset)
            textures = [cls.TextureParam.from_buffer(buffer) for _ in range(texture_count)]
            buffer.seek(parameter_groups_offset)
            parameter_groups = [cls.ParameterGroup.from_buffer(buffer) for _ in range(parameter_groups_count)]
            buffer.seek(variables_offset)
            uniforms = [cls.UniformVariable.from_buffer(buffer) for _ in range(variables_count)]

            buffer.seek(name_offset)
            name = buffer.read_ascii_string()
            buffer.seek(effect_offset)
            effect = buffer.read_ascii_string()
            buffer.seek(technique_offset)
            technique = buffer.read_ascii_string()

        return cls(unk, name, effect, technique, unk2, textures, parameter_groups, uniforms)
