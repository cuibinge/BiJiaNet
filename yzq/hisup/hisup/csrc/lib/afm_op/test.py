import ctypes
import os

# 添加依赖的 DLL 路径到环境变量
os.environ["PATH"] += os.pathsep + r"C:\Windows\System32"
os.environ["PATH"] += os.pathsep + r"D:\program fils\ananconda3\envs\hisup\lib\site-packages\torch\lib"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64"

# 动态加载依赖的 DLL 文件
try:
    ctypes.CDLL(r"C:\Windows\System32\nvcuda.dll")
    print("nvcuda.dll loaded successfully.")
except OSError as e:
    print(f"Error loading nvcuda.dll: {e}")

try:
    ctypes.CDLL(r"D:\program fils\ananconda3\envs\hisup\lib\site-packages\torch\lib\torch_cuda.dll")
    print("torch_cuda.dll loaded successfully.")
except OSError as e:
    print(f"Error loading torch_cuda.dll: {e}")

try:
    ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\cudart64_123.dll")
    print("cudart64_123.dll loaded successfully.")
except OSError as e:
    print(f"Error loading cudart64_123.dll: {e}")