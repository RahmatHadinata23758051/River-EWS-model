import os
import zipfile
from pathlib import Path

def get_long_path(p: Path) -> str:
    # Prepend long path prefix to bypass Windows MAX_PATH limit
    path_str = str(p.resolve())
    prefix = "\\\\?\\"
    if not path_str.startswith(prefix):
        path_str = prefix + path_str
    return path_str

def clean_long_path(path_str: str) -> str:
    # Remove long path prefix for relative paths
    prefix = "\\\\?\\"
    if path_str.startswith(prefix):
        return path_str[4:]
    return path_str

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent # CV-v2/
    zip_path = base_dir / "dataset_colab.zip"
    
    # Target folders to zip
    riwa_dir = base_dir / "Segmentation" / "riwa_v2"
    roboflow_dir = base_dir / "Water detection.v21i.png-mask-semantic"
    train_script = base_dir / "phase1_segmentation" / "src" / "02_train_deeplabv3.py"
    
    print(f"Membuat file ZIP di: {zip_path.name}")
    print("Mohon tunggu, ini akan memakan waktu beberapa menit...\n")
    
    # Open zip file using long path
    with zipfile.ZipFile(get_long_path(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Zip RIWA
        print("Menambahkan dataset RIWA...")
        if riwa_dir.exists():
            long_riwa = get_long_path(riwa_dir)
            for root, dirs, files in os.walk(long_riwa):
                for file in files:
                    file_path = os.path.join(root, file)
                    clean_root = clean_long_path(root)
                    rel_path = Path(clean_root).relative_to(base_dir) / file
                    zipf.write(file_path, str(rel_path).replace("\\", "/"))
                    
        # Zip Roboflow
        print("Menambahkan dataset Roboflow...")
        if roboflow_dir.exists():
            long_robo = get_long_path(roboflow_dir)
            for root, dirs, files in os.walk(long_robo):
                for file in files:
                    file_path = os.path.join(root, file)
                    clean_root = clean_long_path(root)
                    rel_path = Path(clean_root).relative_to(base_dir) / file
                    zipf.write(file_path, str(rel_path).replace("\\", "/"))
                    
        # Zip Script
        print("Menambahkan skrip training...")
        if train_script.exists():
            zipf.write(get_long_path(train_script), str(train_script.relative_to(base_dir)).replace("\\", "/"))
            
    print(f"\n✅ Selesai! File ZIP berhasil dibuat: {zip_path}")
    if zip_path.exists():
        print(f"Ukuran file: {zip_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
