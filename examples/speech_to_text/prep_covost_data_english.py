#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    # COVOST_URL_TEMPLATE = (
    #     "https://dl.fbaipublicfiles.com/covost/"
    #     "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    # )

    # VERSIONS = {2}
    root='/home/rmfrieske/datasets/covost/'
    SPLITS = [ "train"]



    def __init__(
        self,
        root:str,
        split: str,
    ) -> None:
        assert split in self.SPLITS
  


        self.root: Path = Path(root)

        cv_tsv_path = self.root / "combined.tsv"
        assert cv_tsv_path.is_file()

        # Extract features
        # feature_root = out_root / "perturbed_fbank80"
        # feature_root.mkdir(exist_ok=True)

     
   

        df= load_df_from_tsv(cv_tsv_path)
        # print(df)
 
        # df['split']=pd.Categorical(['train']*len(df.index))
        # print(df)
        # if split == "train":
        #     df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        # else:
        #     df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        # print(len(data))
        for e in data:
            
                
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str,  str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence,  speaker_id,
            sample_id)``
        """
        data = self.data[n]
      
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["tgt_text"]
        speaker_id = data["speaker"]
       
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() 
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    # Extract features
    feature_root = root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in CoVoST.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CoVoST(root, split)
     
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, utt_id in tqdm(dataset):
        
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy"
            )
    # Pack features into ZIP
    zip_path = root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
 

    # for split in CoVoST.SPLITS:
    #     manifest = {c: [] for c in MANIFEST_COLUMNS}
    #     dataset = CoVoST(root, split)
    #     print(dataset)
    #     for _, _, src_utt,speaker_id, utt_id in tqdm(dataset):
    #         manifest["id"].append(utt_id)
    #         manifest["audio"].append(audio_paths[utt_id])
    #         manifest["n_frames"].append(audio_lengths[utt_id])
    #         manifest["tgt_text"].append(src_utt )
    #         manifest["speaker"].append(speaker_id)
    #     is_train_split = split.startswith("train")
    #     if is_train_split:
    #         train_text.extend(manifest["tgt_text"])
    #     df = pd.DataFrame.from_dict(manifest)
     
    #     df = filter_manifest_df(df, is_train_split=is_train_split)
    #     save_df_to_tsv(df, root / f"{split}.tsv")
    # Generate vocab
    # vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    # spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}"
    # with NamedTemporaryFile(mode="w") as f:
    #     for t in train_text:
    #         f.write(t + "\n")
    #     gen_vocab(
    #         Path(f.name),
    #         root / spm_filename_prefix,
    #         args.vocab_type,
    #         args.vocab_size
    #     )
    # # Generate config YAML
    # gen_config_yaml(
    #     root,
    #     spm_filename=spm_filename_prefix + ".model",
    #     yaml_filename=f"config.yaml",
    #     specaugment_policy="lb",
    # )
    # Clean up
    # shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root "
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
