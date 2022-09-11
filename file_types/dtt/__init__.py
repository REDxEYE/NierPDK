from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Iterable

from utils.file_utils import IBuffer, FileBuffer, MemoryBuffer


@dataclass(slots=True)
class DTTHeader:
    magic: str
    file_count: int
    file_table_offset: int
    extension_table_offset: int
    name_table_offset: int
    size_table_offset: int
    hashmap_offset: int


@dataclass(slots=True)
class DTTFile:
    name: str
    hash: int
    ext: str
    offset: int
    size: int
    data: Optional[Union[bytes, IBuffer]] = None


class DTTArchive:

    def __init__(self, buffer: IBuffer):
        self._buffer = buffer
        self._files = {}

    def __del__(self):
        self._buffer.close()

    def files(self) -> Iterable[DTTFile]:
        return self._files.values()

    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        header = DTTHeader(buffer.read_ascii_string(4), *buffer.read_fmt('6I'))
        assert header.magic == 'DAT', 'Invalid DTT header'
        file_count = header.file_count
        buffer.seek(header.extension_table_offset)
        extensions = [buffer.read_ascii_string(4) for _ in range(file_count)]

        buffer.seek(header.name_table_offset)
        filename_alignment = buffer.read_uint32()
        filenames = [buffer.read_ascii_string(filename_alignment) for _ in range(file_count)]

        buffer.seek(header.file_table_offset)
        offsets = [buffer.read_uint32() for _ in range(file_count)]

        buffer.seek(header.size_table_offset)
        sizes = [buffer.read_uint32() for _ in range(file_count)]

        self = cls(buffer)
        for i in range(file_count):
            self._files[filenames[i]] = DTTFile(filenames[i], 0, extensions[i], offsets[i], sizes[i])
        return self

    def query_file_by_name(self, name: str) -> Optional[IBuffer]:
        if name in self._files:
            return self._read_file(self._files[name])
        return None

    def query_file_by_id(self, f_id: int) -> Optional[IBuffer]:
        if f_id < len(self._files):
            file = list(self._files.values())[f_id]
            return self._read_file(file)
        return None

    def query_file_by_hash(self, f_hash: int) -> Optional[IBuffer]:
        pass

    def query_file(self, file: DTTFile) -> Optional[IBuffer]:
        return self._read_file(file)

    def _read_file(self, file: DTTFile) -> IBuffer:
        self._buffer.seek(file.offset)
        return MemoryBuffer(self._buffer.read(file.size))


def dtt_from_path(path: Path):
    return dtt_from_buffer(FileBuffer(path, 'rb'))


def dtt_from_buffer(buffer: IBuffer):
    return DTTArchive.from_buffer(buffer)
