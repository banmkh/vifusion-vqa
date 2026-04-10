import argparse
from pathlib import Path
import pandas as pd
# Import các hàm từ module nội bộ
from src.data import convert_json_to_csv, normalize_qa_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess VQA data into CSV files.")
    
    # Argument cho Train
    parser.add_argument("--train-json", type=str, help="Path to training JSON file")
    parser.add_argument("--train-images", type=str, help="Path to training images folder")
    parser.add_argument("--train-csv", type=str, help="Output path for training CSV")
    
    # Argument cho Test
    parser.add_argument("--test-json", type=str, help="Path to test JSON file")
    parser.add_argument("--test-images", type=str, help="Path to test images folder")
    parser.add_argument("--test-csv", type=str, help="Output path for test CSV")

    # Các tùy chọn khác
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip text normalization")
    
    return parser.parse_args()

def preprocess_split(json_path, images_path, csv_path, limit, normalize):
    if not json_path or not images_path or not csv_path:
        return # Bỏ qua nếu không truyền tham số cho split này
    
    print(f"Processing: {json_path} -> {csv_path}")
    
    # Đảm bảo thư mục đầu ra tồn tại
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Chuyển đổi
    convert_json_to_csv(json_path, images_path, csv_path, limit=limit)
    
    # Chuẩn hóa nếu cần
    if normalize:
        df = pd.read_csv(csv_path)
        df = normalize_qa_df(df)
        df.to_csv(csv_path, index=False)
        print(f"Normalized and saved to {csv_path}")

def main() -> None:
    args = parse_args()
    normalize = not args.skip_normalize

    # Xử lý Train split
    preprocess_split(args.train_json, args.train_images, args.train_csv, args.limit, normalize)
    
    # Xử lý Test split
    preprocess_split(args.test_json, args.test_images, args.test_csv, args.limit, normalize)

if __name__ == "__main__":
    main()