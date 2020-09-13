# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
import math
from typing import Optional, Tuple
from tqdm import tqdm

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

from concurrent.futures import ThreadPoolExecutor

def shard_data(path: str, shard_path: str, src_lang: str, tgt_lang: str, n_shards: int):
    # nota bene: src_lang and tgt lang are the suffixes, and path is the prefix
    indexes = [i for i in range(n_shards)]
    new_src_paths = []
    new_tgt_paths = []
    for i in indexes:
        new_src_paths.append(shard_path + "_{}.".format(i) + src_lang)
        new_tgt_paths.append(shard_path + "_{}.".format(i) + tgt_lang)
    for p in new_src_paths + new_tgt_paths:
        assert not os.path.exists(p), f"path: {p} already exists"
    src_fhs = [open(f, mode = "w", encoding="utf-8") for f in new_src_paths]
    tgt_fhs = [open(f, mode="w", encoding="utf-8") for f in new_tgt_paths]
    n_lines = sum(1 for _ in open(path + "." + src_lang, "r"))
    pbar = tqdm(total = n_lines, desc = "sharding the dataset")
    with open(path + "." + src_lang, mode = "r", encoding = "utf-8") as src_fh, \
            open(path + "." + tgt_lang, mode = "r", encoding = "utf-8") as tgt_fh:
        for src_line, tgt_line in zip(src_fh, tgt_fh):
            i = random.sample(indexes, 1)[0]
            src_fhs[i].write(src_line)
            tgt_fhs[i].write(tgt_line)
            pbar.update(1)
    for fh in src_fhs + tgt_fhs:
        fh.close()


class ShardedEpochDatasetIterator:
    def __init__(self, n_shards: int, percent_to_sample: float,
            data_path: str, shard_path: str, extensions: Tuple[str], fields: Tuple[Field], n_threads = 4, n_epochs = 1, **kwargs):
        assert percent_to_sample <= 1.0 and percent_to_sample >= 0.0, "percent to sample needs to be between 0 and 1"
        self.fields = fields
        if not isinstance(self.fields[0], (tuple, list)): 
            self.fields = [("src", fields[0]), ("trg", fields[1])]
        self.n_threads = n_threads
        self.src_shards_paths = [shard_path + "_{}.".format(i) + extensions[0] for i in range(n_shards)]
        self.tgt_shards_paths = [shard_path + "_{}.".format(i) + extensions[1] for i in range(n_shards)]
        for p in self.src_shards_paths + self.tgt_shards_paths:
            assert os.path.exists(p), "uh oh, the sharded data is not available at path {}".format(p)
        self.percent_to_sample = percent_to_sample
        self.get_dataset_kwarg_list = [{"src_path": src_path, "tgt_path": tgt_path, "percent_to_sample": self.percent_to_sample}
                           for src_path, tgt_path in zip(self.src_shards_paths, self.tgt_shards_paths)]
        self.pool = ThreadPoolExecutor(self.n_threads)
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.kwargs = kwargs

    def __iter__(self, reset_n_epochs = None, reset_epoch_count = True):
        if reset_epoch_count:
            self.current_epoch = 0
        if reset_n_epochs:
            self.n_epochs = reset_n_epochs
        return self

    def __next__(self):
        if self.current_epoch < self.n_epochs: 
            self.current_epoch+=1
            return self._get_dataset()
        else:
            raise StopIteration


    def _get_dataset(self):
        examples = []
        pbar = tqdm(total = len(self.get_dataset_kwarg_list), desc = "loading in the sharded data")
        for examples_shard in self.pool.map(self._read_and_sample_wrapper, self.get_dataset_kwarg_list):
            examples.extend([
                data.Example.fromlist([src_line.strip(), tgt_line.strip()], self.fields)
                for src_line, tgt_line in examples_shard]
                            )
            pbar.update(1)
            del examples_shard

        return data.Dataset(examples, self.fields, **self.kwargs)

    def _read_and_sample_wrapper(self, kwargs):
        return self._read_and_sample(**kwargs)

    def _read_and_sample(self, src_path: str, tgt_path: str, percent_to_sample: float):
        with open(src_path, mode = "r", encoding = "utf-8") as src_fh, \
            open(tgt_path, mode = "r", encoding = "utf-8") as tgt_fh:
            src_lines = src_fh.readlines()
            tgt_lines = tgt_fh.readlines()

        line_pairs = list(zip(src_lines, tgt_lines))
        random.shuffle(line_pairs)
        n_samples = math.ceil(percent_to_sample * len(line_pairs))
        result = line_pairs[:n_samples]
        del line_pairs
        return result


def load_data(data_cfg: dict, load_train = True) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    if load_train:

        train_data = TranslationDataset(path=train_path,
                                        exts=("." + src_lang, "." + trg_lang),
                                        fields=(src_field, trg_field),
                                        filter_pred=
                                        lambda x: len(vars(x)['src'])
                                        <= max_sent_length
                                        and len(vars(x)['trg'])
                                        <= max_sent_length)
    else:
        train_data = None

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep

    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field), 
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field), 
                filter_pred=
                lambda x: len(vars(x)['src'])
                <= max_sent_length
                and len(vars(x)['trg'])
                <= max_sent_length)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab, src_field, trg_field


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)
