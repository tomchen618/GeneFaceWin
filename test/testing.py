import datetime
import os
import sys
import subprocess
from pathlib import Path


def copy_files(source, destination):
    try:
        base_dir = os.getcwd()
        base_dir = Path(base_dir).parent
        print("base_dir:", base_dir)
        source_dir = os.path.join(base_dir, source)
        dest_dir = os.path.join(base_dir, destination)
        dest_dir = "D:\\sourcecode\\GeneFacePlusPlus\\checkpoints\\motion2video_nerf\\hewei_head\\codes"
        command = ["xcopy", source_dir, dest_dir, "/s"]
        exclude_file = ""
        print("exclude_file:", exclude_file)
        if exclude_file:
            command.append("/exclude:" + exclude_file)
        print(command)
        command_str = " ".join(command)
        subprocess.run(command_str, shell=True)
    except Exception as error:
        print(error)


def remove_file(*fns):
    if os.name == 'nt':
        for f in fns:
            if os.path.isfile(f):
                os.remove(f)
            else:
                fs = f.split("/")
                base_dir = os.getcwd()
                base_dir = Path(base_dir).parent
                f = base_dir
                for ff in fs:
                    f = os.path.join(f, ff)
                if os.path.isfile(f):
                    os.remove(f)
    else:
        for f in fns:
            subprocess.check_call(f'rm -rf "{f}"', shell=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    c = "tasks"
    code_dir = "checkpoints/motion2video_nerf/hewei_head/codes/test.txt"
    remove_file(code_dir)
    # copy_files(c, code_dir)
