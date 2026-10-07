# LifeLink-AI datasets

## Heart transplant
Primary source requested by the project:
https://www.kaggle.com/datasets/ayyappanmarimuthu/heart-transplant-survival-dataset?resource=download

Because the development environment cannot access Kaggle directly, the package includes `heart/download_kaggle.py`.
Run it once on an internet-connected machine to place the downloaded CSV in `datasets/heart/`. The ML pipeline will automatically prefer that real Kaggle CSV over the bundled demo fallback.

The bundled `heart_transplant_demo.csv` is **synthetic fallback data only** and must not be presented as the Kaggle dataset.

## Kidney donor-recipient matching
The project uses an organ-specific synthetic academic dataset generated from the feature structure and matching classes described in:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11475881/

The published study describes age, sex, ABO, 10 HLA types (A/B/C/DRB1/DQB1) and recipient HLA-antibody specificities, with matching classes including perfect match, acceptable mismatch, ABO mismatch, antibody mismatch and age mismatch. The underlying hospital dataset is not publicly released.

Therefore `kidney_matching_synthetic.csv` is explicitly labeled synthetic and is generated for this academic prototype.
