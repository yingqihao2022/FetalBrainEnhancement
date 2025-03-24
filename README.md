# FetalBrainEnhancement

Here is the public implemention of a fetal brain enhancement model(Submitted to MIDL 2025 Short Paper)

# Procedures
# 1. Preprocessing
    Use generate_dataset_AE.py to generate a json file  for preprocessing.
# 2. Training and Testing
    Use Train.py to train an enhancement model.
    Use Test.py for inference on your dataset. Note that it is ONLY applicable to T2-weighted MRI data.
    Pretrained weights have been provided.
# 3. Postprocessing
    Use transfer.py for postprocessing.
