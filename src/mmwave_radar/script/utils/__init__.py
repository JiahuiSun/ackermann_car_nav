import os
import sys


# 外人调用这个文件夹内的文件，就把这个文件夹添加到path中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
