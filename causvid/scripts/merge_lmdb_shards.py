import argparse
import glob
import os
import lmdb
import numpy as np
from tqdm import tqdm

from causvid.ode_data.create_lmdb_iterative import retrieve_row_from_lmdb, store_arrays_to_lmdb


def open_env(path):
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LMDB shards (with clean_latent/degraded_latent/prompts[/inpainting_mask]) into one LMDB.")
    parser.add_argument("--shards_glob", type=str, required=True, help="Glob pattern to shard lmdbs, e.g. data/.../shard_*.lmdb")
    parser.add_argument("--out_lmdb", type=str, required=True, help="Output merged lmdb path")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.shards_glob))
    assert shard_paths, f"No shards matched: {args.shards_glob}"

    # Estimate map size: sum the data.mdb file sizes from each shard
    total_data_size = sum(os.path.getsize(os.path.join(p, 'data.mdb')) for p in shard_paths if os.path.exists(os.path.join(p, 'data.mdb')))
    map_size = int(total_data_size * 2.0) if total_data_size > 0 else (1024 * 1024 * 1024 * 1024) # Default 1TB
    
    # Clean up previous failed attempts
    if os.path.exists(args.out_lmdb):
        print(f"Removing existing (potentially incomplete) LMDB at {args.out_lmdb}")
        import shutil
        shutil.rmtree(args.out_lmdb)
        
    os.makedirs(os.path.dirname(args.out_lmdb), exist_ok=True)
    print(f"Creating new LMDB with map_size = {map_size/1e9:.2f} GB")
    out_env = lmdb.open(args.out_lmdb, map_size=map_size)

    counter = 0
    clean_shape = None
    degraded_shape = None
    mask_shape = None

    for shard in shard_paths:
        env = open_env(shard)
        with env.begin() as txn:
            cshape = tuple(map(int, txn.get("clean_latent_shape".encode()).decode().split(',')))
            dshape = tuple(map(int, txn.get("degraded_latent_shape".encode()).decode().split(',')))
            mshape_raw = txn.get("inpainting_mask_shape".encode())
            mshape = tuple(map(int, mshape_raw.decode().split(','))) if mshape_raw is not None else None
        num = cshape[0]
        if clean_shape is None:
            clean_shape = cshape
            degraded_shape = dshape
            mask_shape = mshape
        else:
            # ensure latent shapes (except N) are same
            assert clean_shape[1:] == cshape[1:] and degraded_shape[1:] == dshape[1:], "Shard latent shapes mismatch"
            if (mask_shape is None) != (mshape is None) or (mask_shape and (mask_shape[1:] != mshape[1:])):
                raise AssertionError("Shard inpainting mask shapes mismatch")

        for i in tqdm(range(num), desc=f"Merging {os.path.basename(shard)}"):
            clean = retrieve_row_from_lmdb(env, "clean_latent", np.float16, i, shape=cshape[1:])
            degrd = retrieve_row_from_lmdb(env, "degraded_latent", np.float16, i, shape=dshape[1:])
            prompt = retrieve_row_from_lmdb(env, "prompts", str, i)
            if mask_shape is not None:
                mask = retrieve_row_from_lmdb(env, "inpainting_mask", np.float16, i, shape=mask_shape[1:])

            payload = {
                "clean_latent": np.array([clean], dtype=np.float16),
                "degraded_latent": np.array([degrd], dtype=np.float16),
                "prompts": np.array([prompt])
            }
            if mask_shape is not None:
                payload["inpainting_mask"] = np.array([mask], dtype=np.float16)
            store_arrays_to_lmdb(out_env, payload, start_index=counter)
            counter += 1

    # finalize shapes
    with out_env.begin(write=True) as txn:
        txn.put("clean_latent_shape".encode(), ",".join(map(str, (counter,) + clean_shape[1:])).encode())
        txn.put("degraded_latent_shape".encode(), ",".join(map(str, (counter,) + degraded_shape[1:])).encode())
        txn.put("prompts_shape".encode(), ",".join(map(str, (counter, 1))).encode())
        if mask_shape is not None:
            txn.put("inpainting_mask_shape".encode(), ",".join(map(str, (counter,) + mask_shape[1:])).encode())

    print(f"Merged {len(shard_paths)} shards, total {counter} samples -> {args.out_lmdb}")


if __name__ == "__main__":
    main() 