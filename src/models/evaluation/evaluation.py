from src.models.dataset.genomic_dataset import GenomicDataset
from src.models.training.module import TrainModule
from src.models.model.corigami_model import ConvTransModelSmall
from src.models.evaluation.metrics import mse, insulation_corr, distance_stratified_correlation
import torch, numpy as np, argparse, os
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.serialization.add_safe_globals([argparse.Namespace])
import re
from collections import OrderedDict

def _normalize_state_dict_keys(sd, root_prefix_candidates=("model.", "module.", "")):
    """
    Try to strip common wrapper prefixes so keys match the bare model.
    E.g., Lightning checkpoints often store weights under 'model.'.
    """
    # Heuristic: if most keys share a prefix, strip it.
    keys = list(sd.keys())
    for prefix in root_prefix_candidates:
        if prefix and all(k.startswith(prefix) for k in keys):
            return OrderedDict((k[len(prefix):], v) for k, v in sd.items())
    # No single shared prefix; try to drop a leading known segment per key (tolerant)
    cleaned = OrderedDict()
    for k, v in sd.items():
        k_clean = re.sub(r'^(model\.|module\.)', '', k)
        cleaned[k_clean] = v
    return cleaned

def _load_weights_into(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)  # Lightning saves under 'state_dict'; raw torch.save may not
    sd = _normalize_state_dict_keys(sd, root_prefix_candidates=("model.", "module.", "net.", "backbone."))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_weights] Missing keys: {sorted(missing)[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[load_weights] Unexpected keys: {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")
    return model

def init_parser():
    p = argparse.ArgumentParser(description="C.Origami Evaluation Module")
    p.add_argument('--regions-file', required=True)
    p.add_argument('--cool-file', required=True)
    p.add_argument('--fasta-dir', required=True)
    p.add_argument('--genomic-feature', dest='genomic_feature_path', required=True)
    p.add_argument('--num-genom-feat', dest='num_genom_feat', type=int, default=0)
    p.add_argument('--ckpt-path', required=True)
    p.add_argument('--borzoi', action='store_true')
    p.add_argument('--resolution', dest='resolution', type=int, default=400)
    p.add_argument('--n-bins', dest='n_bins', type=int, default=512)
    p.add_argument('--test-chroms', dest='test_chroms', nargs='+', default=["chrX"])
    p.add_argument('--max-offset', type=int, default=None)
    return p.parse_args()


def main():
    args = init_parser()
    if args.num_genom_feat == 0:
        args.genomic_feature_path = None    # <-- ok

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.borzoi)
    if not args.borzoi:
        model = ConvTransModelSmall(mid_hidden=128, num_genomic_features=args.num_genom_feat).to(device)
        model = _load_weights_into(model, args.ckpt_path, device)
    else:
        model = TrainModule.load_from_checkpoint(args.ckpt_path, map_location=device).to(device)
    model.eval()

    # dont wanna mix up n-bins checkpoint
    if args.borzoi:
        ck = model.hparams["args"]
        ck_r, ck_n = getattr(ck, "resolution", 5000), getattr(ck, "n_bins", 105)
        print(f"[eval] checkpoint geometry: resolution={ck_r} n_bins={ck_n}")
        assert (args.resolution, args.n_bins) == (ck_r, ck_n), \
            f"CLI (r={args.resolution}, N={args.n_bins}) != checkpoint (r={ck_r}, N={ck_n})"

    use_pretrained_backbone = bool(args.borzoi)
    corigami_model = not bool(args.borzoi) # clip C.Origami preds for comparison

    overall_mse = []
    overall_pearson = []
    overall_spearman = []
    print('Regions file:', args.regions_file)
    for chrom in args.test_chroms:
        ds = GenomicDataset(
            regions_file_path=args.regions_file,
            cool_file_path=args.cool_file,
            fasta_dir=args.fasta_dir,
            genomic_feature_path=args.genomic_feature_path,
            mode="test",
            test_chroms=[chrom],
            use_pretrained_backbone=use_pretrained_backbone,
            resolution=args.resolution,
            n_bins=args.n_bins,
        )

        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        mse_list = []
        insu_pearson_list = []
        insu_spearman_list = []
        dist_strat_pearson_list = []
        dist_strat_spearman_list = []

        xs, ys = {}, {}
        for batch in tqdm(dl, total=len(dl)):
            test_input = batch["sequence"]
            test_input = test_input.to(device)
            if "features" in batch:
                genom_feat = batch["features"].to(device)
                test_input = torch.cat((test_input, genom_feat), dim=1)

            with torch.no_grad():
                output = model(test_input)
                output = torch.clamp(output, min=0)

            for out, true in zip(output, batch["matrix"]):
                out = out.cpu()  # Move output to CPU
                true = true.cpu()  # Move true matrix to CPU
                if corigami_model:
                    true = true[52:157, 52:157]
                    out = out[52:157, 52:157]

                out = out.squeeze()
                true = true.squeeze()

                r_p, r_s = insulation_corr(out, true, res=args.resolution)
                l_mse = mse(out, true)
                dist_p, dist_s, xs, ys = distance_stratified_correlation(
                    out, true, xs, ys, max_offset=args.max_offset)

                insu_pearson_list.append(r_p)
                insu_spearman_list.append(r_s)
                mse_list.append(l_mse)
                dist_strat_pearson_list.append(dist_p)
                dist_strat_spearman_list.append(dist_s)


        dist_p_mat = np.asarray(dist_strat_pearson_list, dtype=float)
        dist_s_mat = np.asarray(dist_strat_spearman_list, dtype=float)
        xs_flat = {str(d): np.concatenate(v) for d, v in xs.items()}
        ys_flat = {str(d): np.concatenate(v) for d, v in ys.items()}
        np.savez_compressed(
            f"metrics_{chrom}.npz",
            insu_pearson=np.asarray(insu_pearson_list, float),
            insu_spearman=np.asarray(insu_spearman_list, float),
            mse=np.asarray(mse_list, float),
            dist_strat_pearson=dist_p_mat,
            dist_strat_spearman=dist_s_mat,
            **{f"diag_x_{k}": v for k, v in xs_flat.items()},
            **{f"diag_y_{k}": v for k, v in ys_flat.items()},
        )

        overall_mse.extend(mse_list)
        overall_pearson.extend(insu_pearson_list)
        overall_spearman.extend(insu_spearman_list)

        mse_arr = np.asarray(mse_list, dtype=float)
        pearson_arr = np.asarray(insu_pearson_list, dtype=float)
        spearman_arr = np.asarray(insu_spearman_list, dtype=float)

        mse_arr = mse_arr[np.isfinite(mse_arr)]
        pearson_arr = pearson_arr[np.isfinite(pearson_arr)]
        spearman_arr = spearman_arr[np.isfinite(spearman_arr)]

        print(f"{chrom} MSE      mean={mse_arr.mean():.4f}  std={mse_arr.std():.4f}")
        print(f"{chrom} Pearson  mean={pearson_arr.mean():.4f}  std={pearson_arr.std():.4f}")
        print(f"{chrom} Spearman mean={spearman_arr.mean():.4f}  std={spearman_arr.std():.4f}")

    overall_mse = np.asarray(overall_mse, dtype=float)
    overall_pearson = np.asarray(overall_pearson, dtype=float)
    overall_spearman = np.asarray(overall_spearman, dtype=float)

    overall_mse = overall_mse[np.isfinite(overall_mse)]
    overall_pearson = overall_pearson[np.isfinite(overall_pearson)]
    overall_spearman = overall_spearman[np.isfinite(overall_spearman)]

    print(f"Overall MSE      mean={overall_mse.mean():.4f}  std={overall_mse.std():.4f}")
    print(f"Overall Pearson  mean={overall_pearson.mean():.4f}  std={overall_pearson.std():.4f}")
    print(f"Overall Spearman mean={overall_spearman.mean():.4f}  std={overall_spearman.std():.4f}")


if __name__ == "__main__":
    main()
