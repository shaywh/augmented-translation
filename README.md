# Augmented Translation
Custom task for training Fairseq translation models with data augmentation (e.g. BPE-dropout).

This is a template, you'll need to write your preprocessing code in _preprocess method. If needed, you can add arguments to the argument parser in add_args method.

To train translation models with this task, run fairseq-train with --task augmented-translation --user-dir {directory-with-this-task} --unprocessed-data {directory-with-unprocessed-training-data}
