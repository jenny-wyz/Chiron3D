import pandas as pd
from src.models.dataset.genomic_dataset import GenomicDataset


class LoopDataset(GenomicDataset):
    def __init__(self, regions_file_path, cool_file_path, fasta_dir, genomic_feature_path=None,
                 mode="train", val_chroms=None, test_chroms=None, motif="", use_pretrained_backbone=False,
                 use_aug=False, corigami_loops=False, balance=False, matrix_scale=1.0):
        """
        Args:
            regions_file_path (str): Path to the tsv file with the genomic regions that contain the loop regions.
            cool_file_path (str): Path to the .cool file with HiChIP data.
            fasta_dir (str): Directory containing chromosome FASTA files.
            genomic_feature_path (str, optional): Path to the .bw file containing genomic features. Defaults to None.
            mode (str): One of "train", "val", "test" to specify the dataset mode.
            val_chroms (list, optional): List of chromosomes for validation. Defaults to None.
            test_chroms (list, optional): List of chromosomes for testing. Defaults to None.
            encode_motif (str): Which motif to 1hot encode as input. Defaults to empty string.
        """
        super().__init__(regions_file_path, cool_file_path, fasta_dir, genomic_feature_path, mode,
                         val_chroms, test_chroms, use_pretrained_backbone, use_aug,
                         balance=balance, matrix_scale=matrix_scale)
        self.regions_file_path = regions_file_path
        self.cool_file_path = cool_file_path
        self.fasta_dir = fasta_dir
        self.genomic_feature_path = genomic_feature_path
        self.mode = mode
        self.val_chroms = set(val_chroms or [])
        self.test_chroms = set(test_chroms or [])

        self.regions = self._read_regions_file()
        self.cool = self._initialize_cooler()
        self._chrom_set = set(self.cool.chromnames)
        self.filtered_regions = self._filter_regions_by_mode()
        self.encode_motif = motif
        self.bw_files = self._get_bw_files()
        self.dna_channels = 4 if use_pretrained_backbone else 5
        self.resolution = 5000
        self.use_aug = use_aug
        self.corigami_loops = corigami_loops
        self.balance = balance
        self.matrix_scale = matrix_scale

    def _read_regions_file(self):
        """
        Reads the regions file and returns a list of dictionaries.
        """
        df = pd.read_csv(self.regions_file_path, sep='\t')

        return df.to_dict('records')

    def __getitem__(self, idx):
        output = self.filtered_regions[idx].copy()
        output["region_start"] = int(output["region_start"])
        output["region_end"] = int(output["region_end"])
        output["loop_start"] = int(output["loop_start"])
        output["loop_end"] = int(output["loop_end"])

        if self.corigami_loops:
            output["region_start"] -= 260000
            output["region_end"] += 264288

        # Calculate relative loop positions
        output["relative_loop_start"] = int(output['loop_start'] / self.resolution) - int(
            output['region_start'] / self.resolution)
        output["relative_loop_end"] = int(output['loop_end'] / self.resolution) - int(
            output['region_start'] / self.resolution)
        chrom = self._normalize_chrom(output['chr'])

        output = self.build_output(chrom, output)
        return output
        