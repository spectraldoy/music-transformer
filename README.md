# music-transformer

The Music Transformer, or Transformer Decoder with Relative Self-Attention, is a deep learning sequence model that builds upon the Transformer architecture to consider the relative distances between different elements of the sequence rather than / along with their absolute positions in the sequence. I explored my interest in AI-generated music through this project and learned quite a bit about current research in the field of AI in terms of both algorithms and architectures. This repository contains Python scripts to build and train a post-LayerNorm Music Transformer using PyTorch, as well as to generate MIDI files with a trained (or untrained) Music Transformer. 

Refer to [On_the_Music_Transformer.md](https://github.com/spectraldoy/music-transformer/blob/main/On_the_Music_Transformer.md) for details and notes on what makes the Relative Self-Attention mechanism of the Music Transformer so cool.

## Key Dependencies
1. PyTorch ~1.7.1
2. Mido ~1.2.9

## Setting up
Clone, the git repository, cd into it, and install the requirements. Then you're ready to preprocess data, train, and generate music with a Music Transformer.
```shell
git clone https://github.com/spectraldoy/music-transformer
cd ./music-transformer
pip install -r requirements.txt
```

## Preprocessing MIDI Data
Most sequence models require a general upper limit on the length of the sequences being model, it being too computationally or memory expensive to handle longer sequences. So, suppose you have a directory of MIDI files at `.../datapath/`, and would like to convert these files into an event vocabulary that can be trained on, cut these sequences to be less than or equal to an approximate maximum length, `lth`, and store this processed data in a single PyTorch tensor (for use with `torch.utils.data.TensorDataset`) at `.../processed_data.pt`. Running the `preprocessing.py` script as follows:
```shell
python preprocessing.py .../datapath/ .../processed_data.pt lth
```
will translate the MIDI files to the event vocabulary laid out in `vocabulary.py`, tokenize it with functionality from `tokenizer.py`, cut the data to approximately the specified `lth`, augment the dataset by a default set of pitch transpositions and stretches in time, and finally, store the sequences as a single concatenated PyTorch tensor at `.../processed_data.pt`. The cutting is done by randomly generating a number from 0 to `lth`, cutting out that many tokens from the end of the sequence, and padding with `pad_tokens`s to `lth`. The end of the sequence must be kept, so that the model learns to end music. Pitch transpositions and factors for time stretches can also be specified when running the script from the shell (which you can learn about by running `python preprocessing.py -h`).

## Training a Model
Being a ridiculously large model, requiring inordinate amounts of time to train on both GPUs as well as TPUs, the Music Transformer needs to be checkpointed while training. I implemented a deliberate and slightly unwieldy checkpointing mechanism in the `MusicTransformerTrainer` class from `train.py`, to be able to checkpoint while training a Music Transformer. At it's very simplest, given a path to a preprocessed dataset in the form of a PyTorch tensor, `.../preprocessed_data.pt`, and specifying a path at which to checkpoint the model, `.../ckpt_path.pt`, a path at which to save the model, and the number of epochs for which to train the model this current session, `epochs`, running the following:
```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```
will train the model for the specified number of `epochs` on the specified dataset, printing progress messages, will save the model if it finishes training for the specified number of `epochs`, and will checkpoint the optimizer state, learning rate schedule state, model weights, and hyperparameters if a `KeyboardInterrupt` is encountered, or if the model finishes training for the current session. Hyperparameters for the model also be specified when creating a new model, i.e., not loading from a checkpoint (for details on these, run `python train.py -h`). However, if the `--load-checkpoint` flag is also entered:
```shell
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs -l
```
the latest checkpoint stored at `.../ckpt_path.pt` will be loaded, overloading any hyperparameters specified in this script run with the original hyperparameters of the saved model, restoring the model, optimizer, and learning rate schedule states, and continuing training from there. Once training is completed, the model's `state_dict` and `hparams` will be stored in a dictionary and saved at `.../save_path.pt`.

## Generating MIDI Audio

