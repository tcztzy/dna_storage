__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"


from evaluation import EvaluationPipeline

from team0001.coder import Coder


if __name__ == "__main__":
    # "0001" is provided by the competition management party.
    # please see "record.txt" for the process and score records for details.
    coder = Coder(team_id="0001")
    pipeline = EvaluationPipeline(coder=coder, error_free=True)
    pipeline(input_image_path="expected.bmp", output_image_path="obtained.bmp",
             source_dna_path="o.fasta", target_dna_path="p.fasta", random_seed=2023)
    print()
    pipeline = EvaluationPipeline(coder=coder, error_free=False)
    pipeline(input_image_path="expected.bmp", output_image_path="obtained.bmp",
             source_dna_path="o.fasta", target_dna_path="p.fasta", random_seed=2023)
