# music-transformer

The Music Transformer, or Transformer Decoder with Relative Self-Attention, is a deep learning sequence model designed to generate music. It builds upon the Transformer architecture to consider the relative distances between different elements of the sequence rather than / along with their absolute positions in the sequence. I explored my interest in AI-generated music through this project and learned quite a bit about current research in the field of AI in terms of both algorithms and architectures. This repository contains Python scripts to build and train a pre-LayerNorm Music Transformer using PyTorch, as well as to generate MIDI files with a trained (or if you're brave, untrained) Music Transformer. Do create an issue if something does not work as expected.

Refer to [On_the_Music_Transformer.pdf](https://github.com/spectraldoy/music-transformer/blob/main/On_the_Music_Transformer.pdf) (IN PROGRESS) for details and notes on what makes the Relative Self-Attention mechanism of the Music Transformer so cool.

Note that in this README `...` in a path name is used to denote the rest of the full path to a file or directory, not 2 directories above.

## Key Dependencies
1. PyTorch ~1.7.1
2. Mido ~1.2.9

## Setting up
Clone, the git repository, cd into it, and install the requirements. Then you're ready to preprocess MIDI files, as well as train and generate music with a Music Transformer.
```shell
git clone https://github.com/spectraldoy/music-transformer
cd ./music-transformer
pip install -r requirements.txt
```

## Preprocess MIDI Data
Most sequence models require a general upper limit on the length of the sequences being model, it being too computationally or memory expensive to handle longer sequences. So, suppose you have a directory of MIDI files at `.../datapath/` (for instance, any of the folders in the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)), and would like to convert these files into an event vocabulary that can be trained on, cut these sequences to be less than or equal to an approximate maximum length, `lth`, and store this processed data in a single PyTorch tensor (for use with `torch.utils.data.TensorDataset`) at `.../processed_data.pt`. Running the `preprocessing.py` script as follows:
```shell
python preprocessing.py .../datapath/ .../processed_data.pt lth
```
will translate the MIDI files to the event vocabulary laid out in `vocabulary.py`, tokenize it with functionality from `tokenizer.py`, cut the data to approximately the specified `lth`, augment the dataset by a default set of pitch transpositions and stretches in time, and finally, store the sequences as a single concatenated PyTorch tensor at `.../processed_data.pt`. The cutting is done by randomly generating a number from 0 to `lth`, cutting out that many tokens from the end of the sequence, and padding with `pad_tokens`s to the maximum sequence length in the data. The end of the sequence must be kept, so that the model learns to end music. Pitch transpositions and factors of time stretching can also be specified when running the script from the shell (for details, run `python preprocessing.py -h`).

NOTE: THIS SCRIPT WILL NOT WORK PROPERLY FOR MULTI-TRACK MIDI FILES, AND ANY OTHER INSTRUMENTS WILL AUTOMATICALLY BE CONVERTED TO PIANO.
(the reason for this is that I worked only with single-track piano MIDI for this project)

## Train a Music Transformer
Being a ridiculously large model, requiring inordinate amounts of time to train on both GPUs as well as TPUs, the Music Transformer needs to be checkpointed while training. I implemented a deliberate and slightly unwieldy checkpointing mechanism in the `MusicTransformerTrainer` class from `train.py`, to be able to checkpoint while training a Music Transformer. At it's very simplest, given a path to a preprocessed dataset in the form of a PyTorch tensor, `.../preprocessed_data.pt`, and specifying a path at which to checkpoint the model, `.../ckpt_path.pt`, a path at which to save the model, `.../save_path.pt`, and the number of epochs for which to train the model for this session, `epochs`, running the following:
```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```
will train the model for the specified number of `epochs` on the given dataset, printing progress messages, and will checkpoint the optimizer state, learning rate schedule state, model weights, and hyperparameters if a `KeyboardInterrupt` is encountered, anytime a progress message is printed, and when the model finishes training for the specified number of `epochs`. Hyperparameters for the model can also be specified when creating a new model, i.e., not loading from a checkpoint (for details on these, run `python train.py -h`). However, if the `--load-checkpoint` flag is also entered:
```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs -l
```
the latest checkpoint stored at `.../ckpt_path.pt` will be loaded, overloading any hyperparameters specified with the original hyperparameters of the saved model, restoring the model, optimizer, and learning rate schedule states, and continuing training from there. Once training is completed, i.e., the model has been trained for the specified number of `epochs`, the model's `state_dict` and `hparams` will be stored in a Python dictionary and saved at `.../save_path.pt`.

## Generate Music!
Of course the Music Transformer is useless if we can't generate music with it. Given a pretrained Music Transformers's `state_dict` and `hparams` saved at `.../save_path.pt`, and specifying the path at which to save a generated MIDI file, `.../gen_audio.mid`, running the following:
```shell
python generate.py .../save_path.pt .../gen_audio.mid
```
will autoregressively greedy decode the outputs of the Music Transformer to generate a list of token_ids, convert those token_ids back to a MIDI file using functionality from `tokenizer.py`, and will save the output MIDI file at `.../gen_audio.mid`. Parameters for the MIDI generation can also be specified - `'argmax'` or `'categorical'` decode sampling, decode temperature, the number of top_k samples to consider, and the approximate tempo of the generated audio (for more details, run `python generate.py -h`).
