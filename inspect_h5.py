import h5py
import sys


def inspect_h5(path):
    try:
        with h5py.File(path, "r") as f:
            if "model_weights" in f:
                gw = f["model_weights"]
                print(f"Layers: {list(gw.keys())}")
                if "dense_1" in gw:
                    d = gw["dense_1"]
                    print("Dense_1 keys:", list(d.keys()))
                    if "dense_1" in d:
                        print("Dense_1 internal keys:", list(d["dense_1"].keys()))
                        for k in d["dense_1"].keys():
                            print(f"  {k}: {d['dense_1'][k].shape}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_h5(sys.argv[1])
