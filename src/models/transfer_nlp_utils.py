from __future__ import absolute_import, division, print_function
import glob
import logging
import os
import time
import json
import random
import numpy as np
import pandas as pd
from random import sample, seed
import subprocess
import gc
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from transformers import AdamW, get_linear_schedule_with_warmup  # replace
from transformers import glue_compute_metrics as compute_metrics  # replace
from transformers import glue_output_modes as output_modes  # replace
from transformers import glue_processors as processors  # replace
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)  # replace


# make this inherit from Uncertainty class with
# _aggregate_results
# _run_inference
# _infer_one_batch
# CLASSES


class EnsembleInferer:
    def __init__(self, params, dataloader):
        self.dataloader = dataloader
        self.params = params
        self.ens_pred = []
        self.ens_conf = []
        self.obs = []

    def run_inference(self, models):
        model.eval()
        for batch in self.dataloader:
            self._infer_one_batch(batch, models)
        self._aggregate_results()

    def _infer_one_batch(self, batch, models):
        probas = []
        with torch.no_grad():
            xb, mb, _, yb = tuple(t.to(self.params["device"]) for t in batch)
            for i, model in enumerate(models):
                outputs = model["model"](input_ids=xb, attention_mask=mb, labels=yb)
                proba = torch.nn.functional.softmax(outputs[1], dim=1)
                if i == 0:
                    self.nv_conf.append(torch.max(proba, dim=1)[0].cpu().numpy())
                    self.nv_pred.append(proba[:, -1].cpu().numpy())
                probas.append(proba[:, -1].cpu().numpy())
            probas = torch.cat(probas, dim=1)
            assert (
                probas[0, 0] != probas[0, 1]
            ), "Make sure models were trained with different seeds"
            self.ens_pred.append(probas.mean(dim=1).cpu().numpy())
            self.ens_conf.append(probas.std(dim=1).cpu().numpy())
            self.obs.append(yb.cpu().numpy())

    def _aggregate_results(self, pred_cutoff=0.5):
        df = pd.DataFrame(
            {
                "nv_conf": np.concatenate(self.nv_conf),
                "nv_pred": np.concatenate(self.nv_pred),
                "ens_pred": np.concatenate(self.ens_pred),
                "ens_conf": np.concatenate(self.ens_conf),
                "obs": np.concatenate(self.obs),
            }
        )
        df["nv_pred_cls"] = (df["nv_pred"] > pred_cutoff).astype(int)
        df["ens_pred_cls"] = (df["ens_pred"] > pred_cutoff).astype(int)
        self.results = df


class EnsembleTrainer:
    """
    Trains multiple models for inference.`
    """

    def __init__(self, n_models):
        self.n_models = n_models
        self.models = []

    def run(self, params):
        dp = DataProcessor(params)
        for i in range(self.n_models):
            params["seed"] = random.randint(1, 1000)
            set_seed(params["seed"])
            trainer_i = Trainer(params, dp, *get_training_objects(params))
            trainer_i.fit()
            self.models.append({"model": trainer_i.model, "seed": params["seed"]})
            print(f"{i+1}/{self.n_models} models (seed={params['seed']})")


class Experiment:
    """
    Runs an experiment on a given modeling parameter (i.e. data size) and saves results
    """

    def __init__(self, data_sizes, n_seeds, experiment_name):
        self.data_sizes = data_sizesza
        self.n_seeds = n_seeds
        self.results_file = experiment_name + ".json"

    def run(self, params):
        exp_n = len(self.data_sizes) * self.n_seeds
        exp_i = 1
        exp_start = time.time()
        for data_size in self.data_sizes:
            print("\n")
            print("-" * 95)
            print(f"Experiment {exp_i}/{exp_n}: n = {data_size}...")
            print("-" * 95)
            for _ in range(self.n_seeds):
                params["n"] = data_size
                params["seed"] = random.randint(1, 1000)
                set_seed(params["seed"])
                dp = DataProcessor(params)
                trainer = Trainer(params, dp, *get_training_objects(params))
                trainer.fit()
                self._save_results(trainer.results, params)
                exp_i += 1
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
        print(f"Experiments complete {(time.time() - exp_start)/60:5.2f}m")

    def _save_results(self, results, params):
        out_data = {**results, **params}
        with open(self.results_file, "a+") as fout:
            json.dump(out_data, fout)
            fout.write("\n")

    def inspect(self):
        _cols = [
            "model_name",
            "task",
            "bs",
            "lr",
            "num_epochs",
            "max_seq_length",
            "n",
            "seed",
            "train_loss",
            "valid_loss",
            "train_acc",
            "valid_acc",
        ]
        return pd.read_json(self.results_file, lines=True).loc[:, _cols]


def get_training_objects(params):
    """
    Define and return training objects
    """
    config = RobertaConfig.from_pretrained(params["model_name"], num_labels=2)
    model = RobertaForSequenceClassification.from_pretrained(
        params["model_name"], config=config
    )
    model.to(params["device"])
    no_decay = ["bias", "LayerNorm.weight"]
    gpd_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": params["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(gpd_params, lr=params["lr"], eps=params["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["warmup_steps"],
        num_training_steps=params["total_steps"],
    )
    return model, optimizer, scheduler


def print_gpu_obj():
    """Print out gpu objects"""
    count = 0
    for tracked_object in gc.get_objects():
        if torch.is_tensor(tracked_object):
            if tracked_object.is_cuda:
                count += 1
                print(
                    "{} {} {}".format(
                        type(tracked_object).__name__,
                        " pinned" if tracked_object.is_pinned() else "",
                        tracked_object.shape,
                    )
                )
    print(f"\nTHERE ARE {count} OBJECTS ON GPU")


def show_gpu(msg):
    """
    Show gpu utilization
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )

    def to_int(result):
        return int(result.strip().split("\n")[0])

    used = to_int(query("memory.used"))
    total = to_int(query("memory.total"))
    pct = used / total
    print("\n" + msg, f"{100*pct:2.1f}% ({used} out of {total})")


class Trainer:
    """
    Stores the modeling details as well as the handles the training loop.
    """

    def __init__(self, params, data_processor, model, optimizer, scheduler):
        self.params = params
        self.data_processor = data_processor
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(params)
        self.results_file = "training_results.json"

    def save_results(self, results):
        out_data = {**results, **self.params}
        with open(self.results_file, "a+") as fout:
            json.dump(out_data, fout)
            fout.write("\n")
        self.lines += 1

    def inspect(self):
        return pd.read_json(self.results_file, lines=True)

    def on_train_start(self):
        self.train_start = time.time()
        self.data_processor.create_loaders()
        print(
            f'Training {self.params["model_name"]} with lr = {self.params["lr"]},'
            + f'bs = {self.params["bs"]}, n = {self.params["n"]}'
        )
        print("-" * 95)
        self.lc = Losses()
        self.mc = Metrics(self.params["task"], self.params["metric_name"])

    def run_one_epoch(self):
        self.train_all_batches()
        self.valid_all_batches()

    def train_all_batches(self):
        self.epoch_start = time.time()
        self.model.train()
        for batch in self.data_processor.train_dataloader:
            self.train_one_batch(batch)

    def train_one_batch(self, batch):
        xb, mb, _, yb = tuple(t.to(self.params["device"]) for t in batch)
        outputs = self.model(input_ids=xb, attention_mask=mb, labels=yb)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params["max_grad_norm"]
        )
        self.lc.update_batch(outputs, True)
        self.mc.update_batch(outputs, yb, True)
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

    def valid_all_batches(self):
        self.model.eval()
        for batch in self.data_processor.valid_dataloader:
            self.valid_one_batch(batch)

    def valid_one_batch(self, batch):
        xb, mb, _, yb = tuple(t.to(self.params["device"]) for t in batch)
        with torch.no_grad():
            outputs = self.model(input_ids=xb, attention_mask=mb, labels=yb)
            self.lc.update_batch(outputs, False)
            self.mc.update_batch(outputs, yb, False)

    def on_epoch_end(self, epoch):
        train_loss, val_loss = self.lc.update_epoch()
        train_acc, val_acc = self.mc.update_epoch()
        self.early_stopping.check_acc(val_acc, self.model, epoch)
        lr = self.scheduler.get_lr()[0]
        print(
            f'Epoch {epoch:3d}/{self.params["num_epochs"]} '
            + f"| Loss: {train_loss:5.2f} (T) {val_loss:5.2f} (V) "
            + f'| {self.params["metric_name"].upper()}: {train_acc:.3f} (T) {val_acc:.3f} (V) '
            + f"| LR: {lr:02.2} | Time: {time.time() - self.epoch_start:5.2f}s "
        )
        print("-" * 95)
        if self.early_stopping.stop:
            print(
                f"Model stopped improving after epoch {self.early_stopping.best_epoch}"
            )

    def on_train_end(self, save=False, clean_up=False):
        be = self.early_stopping.best_epoch
        results = {
            "train_loss": self.lc.train_losses[be - 1],
            "valid_loss": self.lc.valid_losses[be - 1],
            "train_acc": self.mc.train_accs[be - 1],
            "valid_acc": self.mc.valid_accs[be - 1],
        }
        self.params["best_epoch"] = be
        if save:
            self.save_results(results, params)
        print(f"Training complete {(time.time() - self.train_start)/60:.2f}m")
        if clean_up:
            self.clean_up()
        self.results = results

    def clean_up(self):
        del self.model, self.optimizer, self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        show_gpu("GPU Utilization: ")

    def fit(self):
        self.on_train_start()
        for epoch in range(1, self.params["num_epochs"] + 1):
            self.run_one_epoch()
            self.on_epoch_end(epoch)
            if self.early_stopping.stop:
                break
        self.on_train_end(save=False)


class DataProcessor:
    """
    Preprocess the data, store data loaders and tokenizer
    """

    def __init__(self, params):
        self.params = params
        self.processor = processors[self.params["task"]]()
        self.output_mode = output_modes[self.params["task"]]
        self.label_list = self.processor.get_labels()

    @staticmethod
    def _convert_to_tensors(features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )

    def _load_examples(self, tokenizer, evaluate):
        if evaluate:
            examples = self.processor.get_dev_examples(self.params["data_dir"])
        else:
            examples = self.processor.get_train_examples(self.params["data_dir"])
            if self.params["n"] >= 0:
                examples = sample(examples, self.params["n"])
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=self.label_list,
            max_length=self.params["max_seq_length"],
            output_mode=self.output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )

        return self._convert_to_tensors(features)

    def _define_tokenizer(self):
        return RobertaTokenizer.from_pretrained(
            self.params["model_name"], do_lower_case=True
        )

    def _load_data(self):
        tokenizer = self._define_tokenizer()
        self.train_data = self._load_examples(tokenizer, False)
        self.valid_data = self._load_examples(tokenizer, True)
        self.train_n = len(self.train_data)
        self.valid_n = len(self.valid_data)

    def create_loaders(self):
        self._load_data()
        self.train_dataloader = DataLoader(
            self.train_data, shuffle=True, batch_size=self.params["bs"]
        )
        self.valid_dataloader = DataLoader(
            self.valid_data, shuffle=False, batch_size=2 * self.params["bs"]
        )


class Metrics:
    """Aggregate and store the accuracy for training and validation sets"""

    def __init__(self, task_name, metric_name):
        self.ma_train = MetricAgg(task_name, metric_name)
        self.ma_valid = MetricAgg(task_name, metric_name)
        self.train_accs = []
        self.valid_accs = []

    def update_batch(self, outputs, labels, training):
        if training:
            self.ma_train.update_batch(outputs, labels)
        else:
            self.ma_valid.update_batch(outputs, labels)

    def update_epoch(self):
        self.train_accs.append(self.ma_train.update_epoch())
        self.valid_accs.append(self.ma_valid.update_epoch())
        return self.train_accs[-1], self.valid_accs[-1]


class MetricAgg:
    """Aggregate the accuracy across batches"""

    def __init__(self, task_name, metric_name):
        self.logits = None
        self.preds = None
        self.labels = None
        self.updated_batch = False
        self.task_name = task_name
        self.metric_name = metric_name

    @staticmethod
    def _to_numpy(x):
        return x.detach().cpu().numpy()

    def _first_update_preds_labels(self, logits, labels):
        self.logits = self._to_numpy(logits)
        self.labels = self._to_numpy(labels)
        self.updated_batch = True

    def _update_preds_labels(self, logits, labels):
        self.logits = np.append(self.logits, self._to_numpy(logits), axis=0)
        self.labels = np.append(self.labels, self._to_numpy(labels), axis=0)

    def _get_preds(self):
        self.preds = np.argmax(self.logits, axis=1)

    def update_batch(self, outputs, labels):
        logits = outputs[1]
        if self.updated_batch:
            self._update_preds_labels(logits, labels)
        else:
            self._first_update_preds_labels(logits, labels)

    def update_epoch(self):
        self._get_preds()
        acc = compute_metrics(self.task_name, self.preds, self.labels)[self.metric_name]
        self.updated_batch = False
        return acc


class Losses:
    """Aggregate and store the loss for training and validation sets"""

    def __init__(self):
        self.la_train = LossAgg()
        self.la_valid = LossAgg()
        self.train_losses = []
        self.valid_losses = []

    def update_batch(self, outputs, training):
        if training:
            self.la_train.update_batch(outputs)
        else:
            self.la_valid.update_batch(outputs)

    def update_epoch(self):
        self.train_losses.append(self.la_train.update_epoch())
        self.valid_losses.append(self.la_valid.update_epoch())
        return self.train_losses[-1], self.valid_losses[-1]


class LossAgg:
    """Aggregate the loss across batches"""

    def __init__(self):
        self.n = 0.0
        self.total_loss = 0.0

    def update_batch(self, outputs):
        batch_loss, logits = outputs[:2]
        n_obs = logits.shape[0]
        self.n += n_obs
        self.total_loss += batch_loss.item() * n_obs

    def update_epoch(self):
        loss = self.total_loss / self.n
        self.total_loss = 0.0
        return loss


class EarlyStopping:
    """
    Check whether performance is increasing, stop if not.
    Hat tip to @Bjarten: https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, params, tolerance=0):
        self.patience = params["patience"]
        self.save_name = params["model_name"] + "-checkpoint.pt"
        self.tolerance = tolerance
        self.best_val_acc = None
        self.epochs_no_improve = 0
        self.stop = False
        self.best_epoch = 0

    def check_acc(self, val_acc, model, epoch):

        if self.best_val_acc == None:
            self.best_val_acc = val_acc
            self.epochs_no_improve = 0
            self.save_model(model, epoch)
        elif val_acc > (self.best_val_acc + self.tolerance):
            self.best_val_acc = val_acc
            self.epochs_no_improve = 0
            self.save_model(model, epoch)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.stop = True

    def save_model(self, model, epoch):
        self.best_epoch = epoch
        torch.save(model.state_dict(), self.save_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
