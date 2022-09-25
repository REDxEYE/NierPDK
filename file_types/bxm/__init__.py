from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ...utils.file_utils import IBuffer, FileBuffer


class Node(Dict[str, Optional[str]]):
    def __init__(self):
        super().__init__()
        self.name: Optional[str] = None
        self.value: Optional[str] = None
        self.children: List['Node'] = []

    def __str__(self) -> str:
        value = f": \"{self.value}\"" if self.value else ""
        return f"Node<\"{self.name}\"{value}, {len(self)} attributes>"


class BXM:
    def __init__(self):
        self.flags = 0
        node = Node()
        self.nodes: List[Node] = [node]
        self.root_node: Node = node

    @classmethod
    def from_buffer(cls, buffer: IBuffer) -> 'BXM':
        buffer.set_big_endian()
        magic = buffer.read_ascii_string(4)
        assert magic in ("XML", "BXM"), f"Expected \"XML\" or \"BXM\", got {magic}"
        self = cls()
        self.flags = buffer.read_uint32()
        node_count, data_count, data_size = buffer.read_fmt('2HI')
        nodes_start = buffer.tell()
        buffer.skip(node_count * 8)

        data_offsets = []
        for _ in range(data_count):
            data_offsets.append(buffer.read_fmt('2h'))

        strings_offset = buffer.tell()

        buffer.seek(nodes_start)
        node_info: List[Tuple[int, int, int, int]] = []
        for node_id in range(node_count):
            child_count, first_child_id, attribute_count, data_id = buffer.read_fmt('4H')
            node_info.append((child_count, first_child_id, attribute_count, data_id))
            if node_id == 0:
                continue
            node = Node()
            self.nodes.append(node)

        for (child_count, first_child_id, attribute_count, data_id), node in zip(node_info, self.nodes):
            node.children.extend(self.nodes[first_child_id:first_child_id + child_count])
            name_and_value = data_offsets[data_id]
            with buffer.save_current_offset():
                if name_and_value[0] != -1:
                    buffer.seek(strings_offset + name_and_value[0])
                    node.name = buffer.read_ascii_string()
                if name_and_value[1] != -1:
                    buffer.seek(strings_offset + name_and_value[1])
                    node.value = buffer.read_ascii_string()
                for attr_id in range(attribute_count):
                    attr_info = data_offsets[data_id + 1 + attr_id]
                    if attr_info[0] != -1:
                        buffer.seek(strings_offset + attr_info[0])
                        name = buffer.read_ascii_string()
                    else:
                        name = None
                    if attr_info[1] != -1:
                        buffer.seek(strings_offset + attr_info[1])
                        value = buffer.read_ascii_string()
                    else:
                        value = None
                    if name is None and value is not None:
                        raise Exception("Expected attribute with value to have a name")
                    node[name] = value
        return self


def bxm_from_path(path: Path) -> BXM:
    with FileBuffer(path, 'rb') as buffer:
        return bxm_from_buffer(buffer)


def bxm_from_buffer(buffer: IBuffer) -> BXM:
    return BXM.from_buffer(buffer)
