from src.models.dataset.genomic_dataset import GenomicDataset
ds = GenomicDataset(regions_file_path="data/windows_dm6_C523200_f1024.bed",
                    cool_file_path="data/lbm.800.cool", fasta_dir="data/dmel_chromosomes",
                    mode="train", val_chroms=["chr2L"], test_chroms=["chrX"],
                    use_pretrained_backbone=True, resolution=800, n_bins=654,
                    flank=1024, loop_file="/cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv")
print("windows:", len(ds), " loop rows:", len(ds.loops))
import numpy as np
tot_mass, tot_pos, nz = 0.0, 0, 0
for k in range(len(ds)):
    hm = ds[k]["loop_hm"]
    tot_mass += float(hm.sum()); tot_pos += int((hm > 0.5).sum()); nz += int(hm.sum() > 0)
print("mass/window %.2f   npos/window %.2f   windows with >=1 loop: %d/%d"
      % (tot_mass/len(ds), tot_pos/len(ds), nz, len(ds)))