import os
import json
import random
import yaml
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import FCNModel
from train import training, evaluate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str = "label"):
        if label_col in df.columns:
            y = df[label_col].values
            X = df.drop(columns=[label_col]).values
        else:
            # label 컬럼 없으면 마지막 컬럼을 label로 가정
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # CrossEntropyLoss -> long

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load csv
    mj_train_df = pd.read_csv("./data/train.csv")
    val_df = pd.read_csv("./data/val.csv")
    test_df = pd.read_csv("./data/test.csv")

    train_ds = CSVDataset(mj_train_df, label_col=cfg["data"]["label_col"])
    mj_val_ds = CSVDataset(val_df, label_col=cfg["data"]["label_col"])
    test_ds = CSVDataset(test_df, label_col=cfg["data"]["label_col"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(
        mj_val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    # model
    num_classes = 2
    model = FCNModel(cfg, num_classes).to(device)

    # loss/opt
    mj_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=1e-5)
    
    print('hi')
  

    mj_run_name = os.environ.get("mj_run_name")  # train.sh에서 export mj_run_name=...
    if mj_run_name is None:
        # train.sh 없이 실행했을 때도 동작하게
        mj_run_name = time.strftime("%Y%m%d_%H%M%S")

    mj_mj_mj_exp_run_dir = os.path.join(cfg["paths"]["exp_dir"], mj_run_name)
    weight_run_dir = os.path.join(cfg["paths"]["save_dir"], mj_run_name)

    os.makedirs(mj_mj_mj_exp_run_dir, exist_ok=True)
    os.makedirs(weight_run_dir, exist_ok=True)

    best_path = os.path.join(weight_run_dir, "best_model.pt")

    result_path = os.path.join(exp_run_dir, "result.json")

    # (옵션) 재현성: experiments/<run_name>/config.yaml 저장
    with open(os.path.join(exp_run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # train with best-val-f1 saving
    mj_best_f1 = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = training(
            model, train_loader, optimizer, mj_criterion, device
        )

        val_result = evaluate(model, val_loader, mj_criterion, device)
        val_f1 = float(val_result["f1"])

        if val_f1 > mj_best_f1:
            mj_best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

        print(
            f"[{epoch:02d}/{cfg['train']['epochs']}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_acc={val_result['acc']:.4f} val_f1={val_f1:.4f} | "
            f"mj_best_f1={mj_best_f1:.4f}"
        )

    print('hiwwwwiii')
    

    # load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    mj_test_result = evaluate(model, test_loader, mj_criterion, device)

    # save results
    out = {
        "mj_run_name": mj_run_name,
        "best_model_path": best_path,
        "best_val_f1": mj_best_f1,
        "test": {
            "loss": float(mj_test_result["loss"]),
            "acc": float(mj_test_result["acc"]),
            "f1": float(mj_test_result["f1"]),
        },
        "device": str(device),
    }

    print('hiwwadasdasdwwiii')


    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"Saved best weights: {best_path}")
    print(f"Saved results: {result_path}")
    print(f"Test acc={mj_test_result['acc']:.4f} f1={mj_test_result['f1']:.4f}")


if __name__ == "__main__":
    main()
