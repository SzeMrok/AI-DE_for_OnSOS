from pathlib import Path
import numpy as np
from PIL import Image

output_path = Path("data/inference_samples/noise.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

array = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
image = Image.fromarray(array)

image.save(output_path)

print(f"Saved noise image: {output_path}")
