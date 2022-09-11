from pathlib import Path

from file_types.bxm import bxm_from_path
from file_types.dtt import dtt_from_path
from file_types.wmb import wmb_from_path
from file_types.wta import wta_from_path


def main_dtt():
    if 0:
        root = Path(r'C:\Program Files (x86)\Steam\steamapps\common\NieRAutomata\unpacked\ba')
        for path in root.rglob('*.dat'):
            archive = dtt_from_path(path)
            for file in archive.files():
                with open(f'test_unpack/{file.name}', 'wb') as f:
                    f.write(archive.query_file(file).read())
            print(path)
    archive = dtt_from_path(Path(r"C:\Program Files (x86)\Steam\steamapps\common\NieRAutomata\unpacked\core\coreeff.dtt"))
    print(archive)

def main_wmb():
    if 1:
        root = Path(r'C:\PYTHON_PROJECTS\NierPDK\test_unpack')
        for path in root.rglob('*.wmb'):
            print(path)
            wmb = wmb_from_path(path)
            print(wmb)

    # wmb = wmb_from_path(Path(r'C:\PYTHON_PROJECTS\NierPDK\test_unpack\ba0000.wmb'))
    # print(wmb)


def main_wta():
    if 1:
        root = Path(r'C:\PYTHON_PROJECTS\NierPDK\test_unpack')
        for path in root.rglob('*.wta'):
            print(path)
            wta = wta_from_path(path)
            print(wta)


def main_bxm():
    if 1:
        root = Path(r'C:\PYTHON_PROJECTS\NierPDK\test_unpack')
        for path in root.rglob('*.bxm'):
            print(path)
            bxm = bxm_from_path(path)
            print(bxm)


if __name__ == '__main__':
    main_dtt()
    # main_wmb()
    # main_wta()
    # main_bxm()