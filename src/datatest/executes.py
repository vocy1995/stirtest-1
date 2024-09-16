import subprocess

# 명령어 리스트
commands = [
    "python C:\\Users\\User\\Desktop\\Stir\\STIRMetrics-main\\src\\datatest\\flow2d.py --num_data 10 --showvis 1 --modeltype \"MFT\" --seed 10",
    "python C:\\Users\\User\\Desktop\\Stir\\STIRMetrics-main\\src\\datatest\\flow2d.py --num_data 10 --showvis 1 --modeltype \"MFT\" --seed 12",
    "python C:\\Users\\User\\Desktop\\Stir\\STIRMetrics-main\\src\\datatest\\flow2d.py --num_data 10 --showvis 1 --modeltype \"MFT\" --seed 8",
    "python C:\\Users\\User\\Desktop\\Stir\\STIRMetrics-main\\src\\datatest\\flow2d.py --num_data 10 --showvis 1 --modeltype \"MFT\" --seed 9",
    "python C:\\Users\\User\\Desktop\\Stir\\STIRMetrics-main\\src\\datatest\\flow2d.py --num_data 10 --showvis 1 --modeltype \"MFT\" --seed 11"
]

# 각 명령어 실행
for cmd in commands:
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Command executed successfully: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e}")