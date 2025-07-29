import torch
import time

# GPUが利用可能か確認
if not torch.cuda.is_available():
    print("CUDA (GPU) is not available.")
    exit()

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# 計算用の大きなテンソル（行列）をGPU上に作成
# サイズを大きくすると、より性能差が分かりやすくなります
tensor_size = 10000
a = torch.randn(tensor_size, tensor_size, device=device)
b = torch.randn(tensor_size, tensor_size, device=device)

# ウォームアップ（初回実行は準備に時間がかかるため）
for _ in range(5):
    c = torch.matmul(a, b)

# 時間計測開始
torch.cuda.synchronize() # GPUの処理が終わるのを待つ
start_time = time.time()

# 行列積の計算を10回繰り返す
iterations = 10
for _ in range(iterations):
    c = torch.matmul(a, b)

# 時間計測終了
torch.cuda.synchronize() # GPUの処理が終わるのを待つ
end_time = time.time()

# 結果表示
elapsed_time = end_time - start_time
print(f"Matrix multiplication ({tensor_size}x{tensor_size}) {iterations} times took: {elapsed_time:.4f} seconds.")