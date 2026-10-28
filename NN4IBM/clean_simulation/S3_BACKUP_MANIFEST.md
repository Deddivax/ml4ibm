# NN4IBM curated backup

This S3 snapshot contains project source, scripts, configuration, documentation, notebooks (including Notebook 4's embedded rendered outputs), the Blade geometry, the three six-model Blade graphs, one canonical 2,307,680-row all-72 dataset, and exactly three approved checkpoints.

Included dataset: `outputs/data/exact_original_sdf_lattice_rebuilt_queso_i2_p1_all72.h5`

Included checkpoints:
- `outputs/checkpoints/source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt`
- `outputs/checkpoints/aws_rich_sdf/nn_27_best.pt`
- `outputs/checkpoints/aws_rich_sdf/nn_64_best.pt`

Excluded: all other outputs, logs, experiment artifacts, caches, virtual environments, Git metadata, and all other checkpoints.
