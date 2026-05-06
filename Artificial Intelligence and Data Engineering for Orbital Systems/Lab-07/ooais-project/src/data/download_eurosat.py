from pathlib import Path
from torchvision.datasets import EuroSAT

RAW_DATA_DIR = Path("data/raw")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = EuroSAT(root=str(RAW_DATA_DIR), download=True)
    print(f"Downloaded dataset with {len(dataset)} images")
    print(f"Classes: {dataset.classes}")


if __name__ == "__main__":
    main()