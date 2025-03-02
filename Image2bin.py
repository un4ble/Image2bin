import os
import struct
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

# Zwiększenie limitu wielkości obrazu
Image.MAX_IMAGE_PIXELS = None

# Ścieżka do zapisu pliku (dostosowana do Windowsa)
output_dir = os.path.join(os.getcwd(), "ADSB")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "world_map.bin")

# Wczytaj obraz
im = Image.open("world_map.jpg")
pix = np.array(im, dtype=np.uint8)

# Otwórz plik wyjściowy
with open(output_file, 'wb') as outfile:
    outfile.write(struct.pack('<H', im.size[0]))
    outfile.write(struct.pack('<H', im.size[1]))
    print(f"Rozpoczęto przetwarzanie obrazu {im.filename} o wymiarach {im.size[0]}x{im.size[1]}")

    total_rows = im.size[1]
    progress_step = max(1, total_rows // 20)  # Co 5% postępu
    num_workers = min(cpu_count(), 12)  # Ograniczenie do 16 procesów maksymalnie

    def process_row(y):
        row_data = ((pix[y, :, 0] >> 3) << 11) | ((pix[y, :, 1] >> 2) << 5) | (pix[y, :, 2] >> 3)
        return y, row_data.tobytes()

    with Pool(num_workers) as pool:
        for y, line in sorted(pool.imap_unordered(process_row, range(total_rows))):
            outfile.write(line)
            
            # Komunikaty o postępie
            if y % progress_step == 0 or y == total_rows - 1:
                progress = (y / total_rows) * 100
                print(f"Przetworzono {y}/{total_rows} wierszy ({progress:.1f}%)")

print("Zakończono przetwarzanie! Plik zapisany jako:", output_file)
