# 2023/10/1

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from datapreparing import datapreparing
from denoising_diffusion_pytorch import GaussianDiffusion1D, Trainer1D
from diffusionutils import *
from TimeTransformer import Transformer1, Transformer2, Transformer3, Transformer4, Transformer5
from TimeTransformer.utils import XEDataset
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(20)


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_cli_path(path_like):
    if not path_like:
        return path_like
    path = Path(path_like)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def sample_condition_batches(diffusion_model, condition, device, chunk_size, logger=None, stage_name="sampling"):
    total = len(condition)
    if chunk_size <= 0:
        chunk_size = total
    chunk_size = min(chunk_size, total)

    if logger is not None:
        if chunk_size < total:
            chunk_count = (total + chunk_size - 1) // chunk_size
            logger.info(
                "{}: generating {} samples in {} chunks (chunk size {})".format(
                    stage_name,
                    total,
                    chunk_count,
                    chunk_size,
                )
            )
        else:
            logger.info("{}: generating {} samples in one chunk".format(stage_name, total))

    result = None
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        sampled_seq = diffusion_model.sample(
            torch.as_tensor(condition[start:end], device=device),
            end - start,
        )
        sampled_seq = sampled_seq.detach().cpu().numpy()
        result = sampled_seq if result is None else np.concatenate((result, sampled_seq), axis=0)

    return result


def build_transformer(model_name, d_input, d_model, d_output, num_wl, layernum):
    transformer_cls = {
        "Trans1": Transformer1,
        "Trans2": Transformer2,
        "Trans3": Transformer3,
        "Trans4": Transformer4,
        "Trans5": Transformer5,
    }.get(model_name, Transformer3)

    return transformer_cls(
        d_input=d_input,
        d_model=d_model,
        d_output=d_output,
        num_wl=num_wl,
        N=4,
        layernum=layernum,
        dropout=0.1,
        pe="original",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100001)
    parser.add_argument("--expIndex", type=int, default=888)
    parser.add_argument("--diffusionstep", type=int, default=500)
    parser.add_argument("--denoise", type=str, default="Trans3")
    parser.add_argument("--samplebatchsize", type=int, default=64)
    parser.add_argument("--repeat_num", type=int, default=1)
    parser.add_argument("--train_param_path", type=str, default="../Pretrain/PrepareParams/model_params.npy")
    parser.add_argument("--train_condition_csv", type=str, default="../Pretrain/artifacts/processed_conditions.csv")
    parser.add_argument("--sample_condition_csv", type=str, default="")
    parser.add_argument("--sample_target_param_path", type=str, default="")
    parser.add_argument("--retention_scaler_path", type=str, default="../Pretrain/artifacts/retention_scaler.pkl")
    parser.add_argument("--pec_scaler_path", type=str, default="../Pretrain/artifacts/pec_scaler.pkl")
    parser.add_argument("--wl_vocab_path", type=str, default="../Pretrain/artifacts/wl_vocab.json")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir="TensorBoardLogs/exp{}".format(args.expIndex))

    logger, filename = setup_logger(args.expIndex)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    if_warp = "\n\n" if os.path.getsize(filename) != 0 else ""
    logger.info(if_warp + str(current_time) + ": begin training")

    training_seq, train_condition, sample_condition, gen_target, metadata = datapreparing(
        train_param_path=resolve_cli_path(args.train_param_path),
        train_condition_csv=resolve_cli_path(args.train_condition_csv),
        retention_scaler_path=resolve_cli_path(args.retention_scaler_path),
        pec_scaler_path=resolve_cli_path(args.pec_scaler_path),
        wl_vocab_path=resolve_cli_path(args.wl_vocab_path),
        sample_condition_csv=resolve_cli_path(args.sample_condition_csv) or None,
        sample_target_param_path=resolve_cli_path(args.sample_target_param_path) or None,
        repeat_num=args.repeat_num,
    )

    print("training_seq.shape", training_seq.shape)
    print("condition.shape", train_condition.shape)
    print(args)
    xe_data = XEDataset(training_seq, train_condition)

    denoising_network_choose = args.denoise
    experiment_index = args.expIndex
    diffusion_step = args.diffusionstep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = args.epochs
    batchsize = 64
    sample_times = 1
    transformer_dim = args.modeldim

    denoising_model = build_transformer(
        model_name=denoising_network_choose,
        d_input=training_seq.shape[1],
        d_model=transformer_dim,
        d_output=training_seq.shape[1],
        num_wl=metadata["num_wl"],
        layernum=training_seq.shape[2],
    ).to(device)

    diffusion = GaussianDiffusion1D(
        denoising_model,
        seq_length=training_seq.shape[2],
        timesteps=diffusion_step,
        loss_type="l2",
        objective="pred_v",
        auto_normalize=False,
        beta_schedule="linear",
    ).to(device)

    outputpath = "./Output/exp{}".format(args.expIndex)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    modelsavepath = "./ModelSave/exp{}".format(args.expIndex)
    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)

    trainer = Trainer1D(
        diffusion,
        dataset=xe_data,
        train_batch_size=batchsize,
        train_lr=8e-5,
        train_num_steps=num_epoch,
        gradient_accumulate_every=1,
        save_and_sample_every=50,
        results_folder=modelsavepath,
        ema_decay=0.995,
        amp=False,
        logger=logger,
        condition=sample_condition,
        genTarget=gen_target,
        tbwriter=writer,
        outputpath=outputpath,
        sampleTimes=sample_times,
        sample_batch_size=args.samplebatchsize,
    )
    trainer.train()

    sample_res = None
    for _ in range(sample_times):
        result = sample_condition_batches(
            diffusion,
            sample_condition,
            device,
            args.samplebatchsize,
            logger=logger,
            stage_name="final sampling",
        )
        sample_res = np.expand_dims(result, axis=0) if sample_res is None else np.concatenate((np.expand_dims(result, axis=0), sample_res), axis=0)

    sample_res = np.average(sample_res, axis=0)
    np.save("Output/sampleSeq_RealParams_{}".format(experiment_index), sample_res)
