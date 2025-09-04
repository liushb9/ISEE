if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # 多卡训练相关属性
        self.fabric = None
        self.rank = 0
        self.world_size = 1

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        self.ema_wrapper = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            # 创建EMA包装器
            from diffusion_policy.model.diffusion.ema_model import EMAModel
            self.ema_wrapper = EMAModel(
                model=self.ema_model,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999
            )

        # configure training state
        # 优化器创建移到run方法中，确保模型完全初始化
        self.optimizer = None

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # 创建优化器（确保模型已完全初始化）
        if self.optimizer is None:
            # 使用硬编码的优化器参数，因为配置中已移除optimizer部分
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=1.0e-4,
                betas=(0.95, 0.999),
                eps=1.0e-8,
                weight_decay=1.0e-6
            )
            if self.rank == 0:
                print("✅ 优化器创建成功")

        # 多卡训练设置
        if hasattr(self, 'fabric') and self.fabric is not None:
            # 设置训练
            self.model = self.fabric.setup(self.model)
            self.optimizer = self.fabric.setup_optimizers(self.optimizer)
            
            if self.rank == 0:
                if self.world_size > 1:
                    print(f"✅ 多卡训练模式: rank {self.rank}/{self.world_size}")
                else:
                    print(f"✅ 单卡训练模式: rank {self.rank}/{self.world_size}")
                print(f"   使用设备: {self.fabric.device}")
                print(f"   策略: {self.fabric.strategy}")
        else:
            # 单卡训练
            if self.rank == 0:
                print("⚠️  单卡训练模式 (fabric实例未设置)")

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if self.rank == 0:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        # 多卡训练时使用DistributedSampler
        if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            dataloader_config = copy.deepcopy(cfg.dataloader)
            dataloader_config['shuffle'] = False
            train_dataloader = DataLoader(dataset, sampler=train_sampler, **dataloader_config)
            if self.rank == 0:
                print(f"多卡数据加载: dataset size = {len(dataset)}, sampler size = {len(train_dataloader)}")
        else:
            train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        
        # 使用fabric设置dataloader
        if hasattr(self, 'fabric') and self.fabric is not None:
            train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)
        
        # 只在rank 0上设置验证dataloader
        if hasattr(self, 'fabric') and self.fabric is not None and self.rank == 0:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # 设置normalizer
        if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
            self.model.module.set_normalizer(normalizer)
        else:
            self.model.set_normalizer(normalizer)
            
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = self.ema_wrapper  # 使用已创建的EMA包装器

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        if hasattr(self, 'fabric') and self.fabric is not None:
            device = self.fabric.device
        else:
            device = torch.device(cfg.training.device)
            
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # 多卡训练时设置epoch
                if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                    train_dataloader.sampler.set_epoch(local_epoch_idx)
                    
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                        self.model.module.obs_encoder.eval()
                        self.model.module.obs_encoder.requires_grad_(False)
                    else:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=self.rank != 0,  # 只在rank 0上显示进度条
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                            raw_loss = self.model.module.compute_loss(batch)
                        else:
                            raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        # 使用fabric的backward
                        if hasattr(self, 'fabric') and self.fabric is not None:
                            self.fabric.backward(loss)
                        else:
                            loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                                ema.step(self.model.module)
                            else:
                                ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if self.rank == 0:  # 只在rank 0上记录日志
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                    policy = self.model.module
                else:
                    policy = self.model
                    
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and self.rank == 0:  # 只在rank 0上运行验证
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                if hasattr(self, 'fabric') and self.fabric is not None and self.world_size > 1:
                                    loss = self.model.module.compute_loss(batch)
                                else:
                                    loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and self.rank == 0:  # 只在rank 0上运行采样
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and self.rank == 0:  # 只在rank 0上保存checkpoint
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if self.rank == 0:  # 只在rank 0上记录日志
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
