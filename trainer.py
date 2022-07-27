import gc
import time

import numpy as np
import torch
import torch.nn as nn
from data import CustomCollator, TrainDataset
from ema import EMA
from fgm import FGM
from model import CustomModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from utils import AverageMeter, get_score, timeSince


def train_fn(
    CFG,
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    fgm,
    ema,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            # print(k)
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        #         # adversarial training
        fgm.attack(epsilon=0.1)
        loss_adv = model(inputs).mean()
        # print(f"loss_adv: {loss_adv}")
        loss_adv.backward()
        fgm.restore()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()

        ema.update()

        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
    return losses.avg


def valid_fn(CFG, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    # predictions = np.concatenate(predictions)
    return losses.avg, predictions


def train_loop(folds, fold, CFG, device, OUTPUT_DIR, LOGGER):
    LOGGER.info(f"========== fold: {fold} training ==========")
    tokenizer = CFG.tokenizer
    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds["score"].values

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=CustomCollator(tokenizer),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=CustomCollator(tokenizer),
        drop_last=False,
    )

    # ====================================================
    # model
    # ====================================================
    model = CustomModel(CFG)
    # model = CustomModel(CFG.model, n_vocabs=len(tokenizer))
    torch.save(model.config, OUTPUT_DIR + "config.pth")
    model.to(device)

    # ====================================================
    # freeze embedding
    # ====================================================
    # https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
    # https://stackoverflow.com/questions/71048521/how-to-freeze-parts-of-t5-transformer-model
    # model.base_model.embeddings.requires_grad_(not CFG.freeze_emb)

    if CFG.freeze_emb:
        for n, p in model.named_parameters():
            if "deberta.embeddings" in n:
                p.requires_grad = False  # Actual freezing operation

    # ====================================================
    # optimizer
    # ====================================================
    def get_optimizer_grouped_parameters(model, model_type, lr, weight_decay, llrd):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        group1 = [
            "layer.0.",
            "layer.1.",
            "layer.2.",
            "layer.3.",
            "layer.4.",
            "layer.5.",
            "layer.6.",
            "layer.7.",
        ]
        group2 = [
            "layer.8.",
            "layer.9.",
            "layer.10.",
            "layer.11.",
            "layer.12.",
            "layer.13.",
            "layer.14.",
            "layer.15.",
        ]
        group3 = [
            "layer.16.",
            "layer.17.",
            "layer.18.",
            "layer.19.",
            "layer.20.",
            "layer.21.",
            "layer.22.",
            "layer.23.",
        ]
        # group4=['layer.18.', 'layer.19.', 'layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']
        group_all = [f"layer.{i}." for i in range(24)]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group1)
                ],
                "weight_decay": weight_decay,
                "lr": lr * llrd ** 2,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group2)
                ],
                "weight_decay": weight_decay,
                "lr": lr * llrd,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group3)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            #             {'params': [p for n, p in model.model.deberta.named_parameters()
            #                         if not any(nd in n for nd in no_decay) and any(nd in n for nd in group4)],
            #              'weight_decay': weight_decay, 'lr': lr},
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "lr": lr,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)
                ],
                "weight_decay": 0.0,
                "lr": lr * llrd ** 2,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)
                ],
                "weight_decay": 0.0,
                "lr": lr * llrd,
            },
            {
                "params": [
                    p
                    for n, p in model.model.deberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            #             {'params': [p for n, p in model.model.deberta.named_parameters()
            #                         if any(nd in n for nd in no_decay) and any(nd in n for nd in group4)],
            #              'weight_decay': 0.0, 'lr': lr},
            {
                "params": [
                    p for n, p in model.named_parameters() if model_type not in n
                ],
                "lr": lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer_parameters = get_optimizer_grouped_parameters(
        model,
        model_type="deberta",
        lr=CFG.encoder_lr,
        weight_decay=CFG.weight_decay,
        llrd=CFG.llrd,
    )

    optimizer = AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif cfg.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles,
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # ====================================================
    # FGM
    # ====================================================
    fgm = FGM(model)

    # ====================================================
    # EMA
    # ====================================================
    ema = EMA(model, 0.999)
    ema.register()

    best_score = 0.0

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            CFG,
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            fgm,
            ema,
        )

        ema.apply_shadow()

        # eval
        avg_val_loss, predictions = valid_fn(
            CFG, valid_loader, model, criterion, device
        )

        ema.restore()

        # scoring
        score = get_score(valid_labels, predictions)[0]

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch + 1} - Score: {score:.4f}")

        if best_score < score:
            best_score = score
            LOGGER.info(f"Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            )

    predictions = torch.load(
        OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]
    valid_folds["pred"] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
