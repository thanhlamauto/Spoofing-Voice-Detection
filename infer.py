import argparse
from inference import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake voice detection inference.")

    parser.add_argument("--model", type=str, required=True,
                        choices=["gmm", "cnn", "xgboost", "wav2vec", "svm"],
                        help="Model name to test.")
    parser.add_argument("--num_files", type=int, default=16,
                        help="Number of files to test.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader.")


    args = parser.parse_args()

    test_model(
        model_name=args.model,
        num_files=args.num_files,
        batch_size=args.batch_size
    )