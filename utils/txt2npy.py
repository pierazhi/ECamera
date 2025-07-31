import numpy as np
import argparse
import os

def convert_txt_to_npy(txt_path, npy_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Input file '{txt_path}' not found.")

    # Load as [t, x, y, p]
    data = np.loadtxt(txt_path, dtype=np.float64)

    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in input, got {data.shape[1]}. Check your file format.")

    # Convert polarity from [0, 1] → [-1, 1]
    data[:, 3] = 2 * data[:, 3] - 1

    # Reorder to [x, y, t, p]
    data_reordered = data[:, [1, 2, 0, 3]]

    # Save in correct format
    np.save(npy_path, data_reordered)
    print(f"✅ Converted '{txt_path}' to '{npy_path}' with {data_reordered.shape[0]} events in [x, y, t, p] format.")

def main():
    parser = argparse.ArgumentParser(description="Convert event text file [t,x,y,p] to .npy [x,y,t,p] with polarity in {-1,1}.")
    parser.add_argument("input_txt", help="Path to the input .txt file")
    parser.add_argument("output_npy", help="Path to the output .npy file")
    args = parser.parse_args()

    convert_txt_to_npy(args.input_txt, args.output_npy)

if __name__ == "__main__":
    main()

