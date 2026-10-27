import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]


def print_npz():
    npz_path = ROOT_PATH / Path("data/embeddings/000001_2023_平安银行_chunk.npz")
    with np.load(npz_path) as data:
        print("内部数组key列表：", list(data.keys()))
        print("n =", data["n"])
        print("dense shape:", data["dense"].shape, data["dense"].dtype)
        print("sp_idx shape:", data["sp_idx"].shape, data["sp_idx"].dtype)
        print("sp_val shape:", data["sp_val"].shape, data["sp_val"].dtype)
        print("sp_offsets shape:", data["sp_offsets"].shape, data["sp_offsets"].dtype)

        # 取出第一个chunk的向量
        offsets = data["sp_offsets"]
        lo, hi = offsets[0], offsets[1]
        print("\n第0号chunk稀疏向量：")
        print("indices：", data["sp_idx"][lo:hi])
        print("values：", data["sp_val"][lo:hi])


if __name__ == "__main__":
    print_npz()
