model_paths = list((Path(tensorboard.save_dir) / LOG_NAME).glob("version_*/checkpoints/*"))
model_paths