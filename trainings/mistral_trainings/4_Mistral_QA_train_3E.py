import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import StateDictType
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
import wandb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
import sys
import platform
from datetime import datetime
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.distributed import all_reduce, ReduceOp
from torch.distributed.checkpoint import (
    FileSystemReader, 
    FileSystemWriter,
    load_state_dict,
    save_state_dict
)
from transformers.trainer_utils import enable_full_determinism
import logging

enable_full_determinism(42)  # Для воспроизводимости

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#os.environ["WANDB_API_KEY"] = "2df04fb1a8c840899ff049c5e363c23e36bfbf55"  
#os.environ["WANDB_MODE"] = "online"

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def get_auto_wrap_policy():
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MistralDecoderLayer}  
    )

def calculate_accuracy(logits, labels, tokenizer):
    predictions = torch.argmax(logits, dim=-1)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_predictions = predictions[..., :-1].contiguous()
    mask = shifted_labels != -100
    mask = mask & (shifted_labels != tokenizer.pad_token_id)
    correct = (shifted_predictions == shifted_labels) & mask
    return correct.float().sum() / mask.float().sum()

def evaluate(model, loader, scaler, rank, world_size, tokenizer):
    # Валидация вызывается на всех ранках

    dist.barrier()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    pbar = tqdm(total=len(loader), desc="Validating", leave=False, disable=rank != 0) if rank == 0 else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                accuracy = calculate_accuracy(outputs.logits, batch['input_ids'],tokenizer)
            
            batch_size = batch['input_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_accuracy += accuracy.item() * batch_size
            total_samples += batch_size
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{total_loss/total_samples:.3f}",
                    "acc": f"{(total_accuracy/total_samples)*100:.1f}%"
                })
    
    if pbar:
        pbar.close()
    
    # Синхронизация метрик между устройствами
    avg_loss = torch.tensor(total_loss / total_samples).cuda()
    avg_acc = torch.tensor(total_accuracy / total_samples).cuda()
    
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    
    return {
        'loss': avg_loss.item() / world_size,
        'accuracy': avg_acc.item() / world_size
    }


def format_prompt(row):
    from ast import literal_eval
    options = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(literal_eval(row['options']))])
    return f"""<s>[INST] Analyze the question and select the correct answer. Answer should be a single uppercase letter. Question:{row['question']}\nOptions:\n{options} [/INST] Answer:{row['answer']}</s>"""

def prepare_datasets(train_file, val_file, test_file, tokenizer):
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_df = pd.read_csv(train_file, sep="\t")
    val_df   = pd.read_csv(val_file, sep="\t")
    test_df  = pd.read_csv(test_file, sep="\t")
    
    # Функция для создания Dataset
    def create_dataset(df_split):
        formatted = df_split.apply(format_prompt, axis=1)
        dataset = Dataset.from_pandas(pd.DataFrame({'text': formatted}), preserve_index=False)
        return dataset.map(
            lambda ex: tokenizer(
                ex["text"],
                truncation=True,
                max_length=768,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=False
            ),
            remove_columns=["text"],
            num_proc=4
        )
    
    return create_dataset(train_df), create_dataset(val_df), create_dataset(test_df)    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Mistral-Small-24B-Instruct-2501")
    #parser.add_argument("--data_path", default="mmlu_pro_stem.tsv") Разделил на три
    parser.add_argument("--train_path", default="easy_window_train.tsv")
    parser.add_argument("--valid_path", default="easy_window_valid.tsv")
    parser.add_argument("--test_path", default="combined_window_test.tsv")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", default="mistral_finetuned_light")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--offload", action="store_true")
    
    args = parser.parse_args()
    
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        log_file = os.path.join(args.save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("=== Environment Info ===")
        venv_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
        if venv_path:
            env_type = "Conda" if 'conda' in venv_path.lower() else "Virtualenv"
            logging.info(f"Environment: {env_type} ({venv_path})")
        logging.info(f"Python: {platform.python_version()}")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info("Starting training...")

        #wandb.init(project="qa_mistral_train", entity="danielvyazhev-hse", config=vars(args))

    # Инициализация модели
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    # Настройка FSDP
    model = FSDP(
        model,
        auto_wrap_policy=get_auto_wrap_policy(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Наша стратегия для шардирования
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=args.offload),
        limit_all_gathers=True
    )


    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    #df = pd.read_csv(args.data_path, sep='\t').head(300)
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_file=args.train_path,
        val_file=args.valid_path,
        test_file=args.test_path,
        tokenizer=tokenizer
        )
    
    num_workers = min(os.cpu_count() // world_size, 8)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )


    def create_loader(dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=DistributedSampler(dataset, shuffle=shuffle),
            collate_fn=data_collator,
            pin_memory=False,
            num_workers=num_workers
        )

    train_loader = create_loader(train_dataset, shuffle=True)
    test_loader = create_loader(test_dataset)

    # Отдельный лоадер для валидации
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_dataset, shuffle=False),  
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )


    # Оптимизатор 
    optimizer = AdamW(model.parameters(), lr=args.lr, fused=True)
    scaler = GradScaler(enabled=False) # Для маленьких ресурсов можно включить
    
    if rank == 0:
        sample_batch = next(iter(train_loader))
        input_ids = sample_batch['input_ids'][0]
        
        # Пример для отладки
        original_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        clean_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        logging.info("=== Пример входных данных ===")
        logging.info("Original text with special tokens:")
        logging.info(original_text)
        logging.info("Clean text:")
        logging.info(clean_text)
        

    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)


        if rank == 0:
            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs}",
                dynamic_ncols=True,
                postfix={
                    "loss": "init", 
                    "acc": "0.00%",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
                }
            )
            logging.info(f"Starting Epoch {epoch+1}/{args.epochs}")
            
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_samples = 0
        

        for batch_idx, batch in enumerate(train_loader):
            try:
                batch = {k: v.to(rank, non_blocking=True) for k, v in batch.items()}

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation
                    accuracy = calculate_accuracy(outputs.logits, batch['input_ids'],tokenizer)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    # Если GradScaler включён – выполнить unscale, шаг оптимизатора и обновление скейлера
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Обновление метрик и т.д.
                batch_size = batch['input_ids'].size(0)
                total_train_loss += loss.item() * batch_size
                total_train_acc += accuracy.item() * batch_size
                total_samples += batch_size

                if rank == 0 and (batch_idx % 10 == 0 or batch_idx == len(train_loader)-1):
                    current_loss = total_train_loss / (total_samples + 1e-7)
                    current_acc = total_train_acc / (total_samples + 1e-7)
                    pbar.set_postfix({
                        "loss": f"{current_loss * args.gradient_accumulation:.3f}",
                        "acc": f"{current_acc * 100:.2f}%",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
                    })
                    pbar.update(batch_idx - pbar.n)  
                    
                    logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {current_loss * args.gradient_accumulation:.3f}, Acc = {current_acc * 100:.2f}%")
                
                if (batch_idx + 1) % args.log_interval == 0:
                    train_metrics = {
                        'loss': torch.tensor(total_train_loss / total_samples).cuda(),
                        'accuracy': torch.tensor(total_train_acc / total_samples).cuda()
                    }
                    dist.all_reduce(train_metrics['loss'], op=dist.ReduceOp.SUM)
                    dist.all_reduce(train_metrics['accuracy'], op=dist.ReduceOp.SUM)
                    if rank == 0:
                        train_loss = train_metrics['loss'].item() / world_size
                        train_acc = train_metrics['accuracy'].item() / world_size
                        logging.info(f"Log Interval: Avg Loss = {train_loss:.4f}, Avg Acc = {train_acc*100:.2f}%")

                    total_train_loss = 0.0
                    total_train_acc = 0.0
                    total_samples = 0
            
        

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logging.error(f"OOM error on rank {rank}, skipping batch {batch_idx}")
                    torch.cuda.empty_cache()
                    if rank == 0:
                        pbar.write(f"Skipped batch {batch_idx} due to OOM")
                    continue
                else:
                    if rank == 0:
                        pbar.close()
                    raise
 
        if rank == 0:
            pbar.close()
            logging.info(f"Epoch {epoch+1} completed.")
            pbar = None

        # Вызов валидации на всех ранках
        val_metrics = evaluate(model, val_loader, scaler, rank, world_size, tokenizer)
        if rank == 0:
            logging.info(f"Validation Results for Epoch {epoch+1}: Loss = {val_metrics['loss']:.4f}, Accuracy = {val_metrics['accuracy']*100:.2f}%")

            # wandb.log({
            #     "val/loss": val_metrics['loss'],
            #     "val/accuracy": val_metrics['accuracy']
            # })

        # Синхронизация после валидации, чтобы все ранки завершили вычисления
        dist.barrier()
        epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        if epoch == args.epochs - 1:
    
            config = AutoConfig.from_pretrained(args.base_model)
            config.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
            
            if rank == 0:

                # logging.info("Saving full model (pytorch_model.bin) via torch.save")
                # torch.save(cpu_state_dict, f"{epoch_dir}/pytorch_model.bin")
                
                logging.info("Saving model for inference via inference_model.save_pretrained")
                # Перекидываем модель на CPU для сохранения
                inference_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )

                # Загружаем веса и сохраняем
                inference_model.load_state_dict(cpu_state_dict)
                inference_model.save_pretrained(
                    epoch_dir,
                    max_shard_size="10GB",
                    safe_serialization=True
                )
                logging.info(f"Full model saved for inference in {epoch_dir}")

            torch.cuda.empty_cache()
            if 'cpu_state_dict' in locals():
                del cpu_state_dict
            if 'inference_model' in locals():
                del inference_model

            dist.barrier()

        # В случае если будем сохранять чекпоинты
        # else:
        #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        #     with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        #         model_state = model.state_dict()

        #     torch.save(
        #         {'model_state_dict': model_state},
        #         os.path.join(epoch_dir, "fsdp_checkpoint.pt")
        #     )
        #     logging.info(f"Checkpoint saved for Epoch {epoch+1} at {epoch_dir}")

        # torch.cuda.empty_cache()

    # Финальное тестирование — вызывается на всех ранках
    test_metrics = evaluate(model, test_loader, scaler, rank, world_size, tokenizer)
    if rank == 0:
        logging.info(f"Final Test Results: Loss = {test_metrics['loss']:.4f}, Accuracy = {test_metrics['accuracy']*100:.2f}%")

        # wandb.log({
        #     "test/loss": test_metrics['loss'],
        #     "test/accuracy": test_metrics['accuracy']
        # })
    
    cleanup()


if __name__ == "__main__":
    main()
