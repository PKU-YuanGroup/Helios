import os
import pickle
import random
from collections import defaultdict

import torch
from einops import rearrange
from torch.utils.data import Dataset, Sampler


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        gan_folders=None,
        ode_folders=None,
        text_folders=None,
        is_use_gt_history=False,
        return_secondary=False,
        force_rebuild=False,
        single_res=True,
        single_length=True,
        single_num_frame=81,
        single_height=384,
        single_width=640,
        seed=42,
    ):
        self.is_use_gt_history = is_use_gt_history
        self.return_secondary = return_secondary
        self.force_rebuild = force_rebuild
        self.base_seed = seed
        self._epoch = 0

        self.single_res = single_res
        self.single_length = single_length
        self.single_num_frame = single_num_frame
        self.single_height = single_height
        self.single_width = single_width

        self.gan_samples = self._init_samples(gan_folders, "gan")
        self.ode_samples = self._init_samples(ode_folders, "ode")
        self.text_samples = self._init_samples(text_folders, "text")

        self._align_sample_counts()

    def _init_samples(self, folders, data_type):
        if folders is None:
            return []

        folders = [folders] if isinstance(folders, str) else folders
        samples = []

        for folder in folders:
            cache_file = os.path.join(folder, f"{data_type}_dataset_cache.pkl")
            folder_samples = self._process_folder(folder, cache_file, data_type)
            samples.extend(folder_samples)

        return samples

    def _align_sample_counts(self, is_log=True):
        lengths = {"gan": len(self.gan_samples), "ode": len(self.ode_samples), "text": len(self.text_samples)}

        non_empty_lengths = {k: v for k, v in lengths.items() if v > 0}
        if not non_empty_lengths:
            return
        max_length = max(non_empty_lengths.values())

        if is_log:
            print(f"\nAligning sample counts to max: {max_length}")
            print(f"Original counts - GAN: {lengths['gan']}, ODE: {lengths['ode']}, TEXT: {lengths['text']}")

        random.seed(self.base_seed)

        if self.gan_samples and len(self.gan_samples) < max_length:
            self.gan_samples = self._expand_samples(self.gan_samples, max_length, "GAN")

        if self.ode_samples and len(self.ode_samples) < max_length:
            self.ode_samples = self._expand_samples(self.ode_samples, max_length, "ODE")

        if self.text_samples and len(self.text_samples) < max_length:
            self.text_samples = self._expand_samples(self.text_samples, max_length, "TEXT")

        if is_log:
            print(
                f"Aligned counts - GAN: {len(self.gan_samples)}, ODE: {len(self.ode_samples)}, TEXT: {len(self.text_samples)}\n"
            )

    def _expand_samples(self, samples, target_length, data_type):
        original_length = len(samples)
        expanded_samples = samples.copy()

        while len(expanded_samples) < target_length:
            random_sample = random.choice(samples)
            expanded_samples.append(random_sample)

        print(f"{data_type}: Expanded from {original_length} to {len(expanded_samples)} samples")
        return expanded_samples

    def _process_folder(self, folder, cache_file, data_type):
        if self.force_rebuild or not os.path.exists(cache_file):
            # if os.path.exists(cache_file):
            #     os.remove(cache_file)
            print(f"{data_type.upper()}: Building metadata cache for folder: {folder}")
            folder_samples = self._build_folder_metadata(folder, data_type)

            if not self.force_rebuild:
                print(f"{data_type.upper()}: Saving metadata cache for folder: {folder}")
                with open(cache_file, "wb") as f:
                    pickle.dump({"samples": folder_samples}, f)

            print(f"{data_type.upper()}: Cached {len(folder_samples)} samples from {folder}")
        else:
            print(f"{data_type.upper()}: Loading cached metadata from: {folder}")
            with open(cache_file, "rb") as f:
                folder_samples = pickle.load(f)["samples"]
            print(f"{data_type.upper()}: Loaded {len(folder_samples)} samples from cache: {folder}")

        return folder_samples

    def _build_folder_metadata(self, folder, data_type):
        feature_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        samples = []

        print(f"{data_type.upper()}: Processing {len(feature_files)} files in {folder}...")
        for i, feature_file in enumerate(feature_files):
            if i % 10000 == 0:
                print(f"  {data_type.upper()}: Processed {i}/{len(feature_files)} files")

            feature_path = os.path.join(folder, feature_file)

            # TODO hard code here now
            if data_type == "gan":
                parts = feature_file.split("_")
                num_frame = int(parts[-3])
                height = int(parts[-2])
                width = int(parts[-1].replace(".pt", ""))

                if self.is_use_gt_history:
                    if (height, width) not in [(self.single_height, self.single_width)]:
                        continue
                else:
                    if (num_frame, height, width) not in [
                        (self.single_num_frame, self.single_height, self.single_width)
                    ]:
                        continue

            samples.append(
                {
                    "uttid": os.path.splitext(os.path.basename(feature_file))[0],
                    "dataset_name": folder.rstrip("/"),
                    "file_path": feature_path,
                }
            )

        return samples

    def prepare_stage1_latent(self, vae_latent, idx, base_vae_latent=None, return_secondary=False):
        self.is_keep_x0 = (True,)
        self.history_sizes = [16, 2, 1]
        self.num_rollout_sections = 9

        source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

        x0_latent = None
        if self.is_keep_x0:
            x0_latent = source_latent[0, :, :1, :, :].clone()
        total_sections = source_latent.shape[0]
        latent_window_size = source_latent.shape[2]
        history_window_size = sum(self.history_sizes)
        section_size = history_window_size + latent_window_size

        temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
        zero_padding_source = torch.zeros(
            temp_source_latent.shape[0],
            history_window_size,
            temp_source_latent.shape[2],
            temp_source_latent.shape[3],
            device=temp_source_latent.device,
            dtype=temp_source_latent.dtype,
        )
        continue_source_latent = torch.cat([zero_padding_source, temp_source_latent], dim=1)

        temp_vae_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
        zero_padding_vae = torch.zeros(
            temp_vae_latent.shape[0],
            history_window_size,
            temp_vae_latent.shape[2],
            temp_vae_latent.shape[3],
            device=temp_vae_latent.device,
            dtype=temp_vae_latent.dtype,
        )
        continue_vae_latent = torch.cat([zero_padding_vae, temp_vae_latent], dim=1)

        sample_seed = self.base_seed + self._epoch * 1000000 + idx
        choice_idx = torch.randint(
            0, total_sections, (1,), generator=torch.Generator().manual_seed(sample_seed)
        ).item()
        if choice_idx == 0 and x0_latent is not None:
            x0_latent = torch.zeros_like(x0_latent)

        start_indice = choice_idx * latent_window_size
        end_indice = start_indice + section_size

        history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
        target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]

        x0_latent_2 = None
        history_latent_2 = None
        target_latent_2 = None
        if return_secondary:
            sample_seed_2 = self.base_seed + self._epoch * 1000000 + idx + 999999
            choice_idx_2 = torch.randint(
                0, total_sections, (1,), generator=torch.Generator().manual_seed(sample_seed_2)
            ).item()

            x0_latent_2 = None
            if self.is_keep_x0:
                x0_latent_2 = source_latent[0, :, :1, :, :].clone()
                if choice_idx_2 == 0:
                    x0_latent_2 = torch.zeros_like(x0_latent_2)

            start_indice_2 = choice_idx_2 * latent_window_size
            end_indice_2 = start_indice_2 + section_size

            history_latent_2 = continue_source_latent[:, start_indice_2 : start_indice_2 + history_window_size, :, :]
            target_latent_2 = continue_vae_latent[:, start_indice_2 + history_window_size : end_indice_2, :, :]

        return (x0_latent, history_latent, target_latent), (x0_latent_2, history_latent_2, target_latent_2)

    def set_epoch(self, epoch):
        self._epoch = epoch
        random.seed(self.base_seed + epoch)
        self._align_sample_counts(is_log=False)

    def __len__(self):
        return max(len(self.gan_samples), len(self.ode_samples), len(self.text_samples))

    def __getitem__(self, idx):
        while True:
            try:
                output_dict = {}

                if self.gan_samples:
                    gan_sample = self.gan_samples[idx]
                    gan_feature = torch.load(gan_sample["file_path"], map_location="cpu", weights_only=False)
                    if self.is_use_gt_history:
                        (
                            (x0_latent, history_latent, target_latent),
                            (x0_latent_2, history_latent_2, target_latent_2),
                        ) = self.prepare_stage1_latent(
                            gan_feature["vae_latent"],
                            idx,
                            return_secondary=self.return_secondary,
                        )
                        output_dict.update(
                            {
                                "gan_uttid": gan_sample["uttid"],
                                "gan_dataset_name": gan_sample["dataset_name"],
                                "gan_vae_latents": target_latent,
                                "gan_x0_latents": x0_latent,
                                "gan_history_latents": history_latent,
                                "gan_vae_latents_2": target_latent_2,
                                "gan_x0_latents_2": x0_latent_2,
                                "gan_history_latents_2": history_latent_2,
                                "gan_prompt_raws": gan_feature["prompt_raw"],
                                "gan_prompt_embeds": gan_feature["prompt_embed"],
                            }
                        )
                    else:
                        output_dict.update(
                            {
                                "gan_uttid": gan_sample["uttid"],
                                "gan_dataset_name": gan_sample["dataset_name"],
                                "gan_vae_latents": gan_feature["vae_latent"],
                                "gan_prompt_raws": gan_feature["prompt_raw"],
                                "gan_prompt_embeds": gan_feature["prompt_embed"],
                            }
                        )
                    gan_sample = None
                    gan_feature = None
                    del gan_sample
                    del gan_feature

                if self.ode_samples:
                    ode_sample = self.ode_samples[idx]
                    ode_feature = torch.load(ode_sample["file_path"], map_location="cpu", weights_only=False)
                    output_dict.update(
                        {
                            "ode_uttid": ode_sample["uttid"],
                            "ode_dataset_name": ode_sample["dataset_name"],
                            "ode_latent_window_size": ode_feature["latent_window_size"],
                            "ode_latents": ode_feature["ode_latents"],
                            "ode_prompt_raws": ode_feature["prompt_raw"],
                            "ode_prompt_embeds": ode_feature["prompt_embed"][0],
                        }
                    )
                    ode_sample = None
                    ode_feature = None
                    del ode_sample
                    del ode_feature

                if self.text_samples:
                    text_sample = self.text_samples[idx]
                    text_feature = torch.load(text_sample["file_path"], map_location="cpu", weights_only=False)
                    output_dict.update(
                        {
                            "text_uttid": text_sample["uttid"],
                            "text_dataset_name": text_sample["dataset_name"],
                            "text_prompt_raws": text_feature["prompt_raw"],
                            "text_prompt_embeds": text_feature["prompt_embed"],
                        }
                    )
                    text_sample = None
                    text_feature = None
                    del text_sample
                    del text_feature

                return output_dict

            except Exception as e:
                idx = random.randint(0, len(self) - 1)
                print(f"Error loading sample at idx {idx}, retrying... Error: {e}")


class BucketedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        dataset_sampling_ratios={},
        drop_last=False,
        shuffle=True,
        seed=42,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self._epoch = 0

        # Distributed parameters
        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        """
        Shard indices across SP groups.
        Each SP group gets a disjoint subset of the data.
        """
        if self.num_sp_groups == 1:
            return indices

        # Convert to tensor if it's a list
        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long)
        else:
            indices_tensor = indices

        # Pad indices if necessary to make it divisible by num_sp_groups
        total_size = len(indices_tensor)
        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
            else:
                # drop_last must truncate here; the divisible case needs no truncation.
                truncate_size = (total_size // self.num_sp_groups) * self.num_sp_groups
                indices_tensor = indices_tensor[:truncate_size]

        # Shard: each SP group gets every num_sp_groups-th element
        sp_group_indices = indices_tensor[self.ith_sp_group :: self.num_sp_groups]

        return sp_group_indices.tolist()

    def __iter__(self):
        # Use epoch-level seed for reproducibility
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        # Get all indices
        all_indices = list(range(len(self.dataset)))

        # Global shuffle before sharding (important for distributed consistency)
        if self.shuffle:
            perm = torch.randperm(len(all_indices), generator=self.generator).tolist()
            all_indices = [all_indices[i] for i in perm]

        # Shard indices for this SP group
        sp_group_indices = self._shard_indices_for_sp_group(all_indices)

        # Create batches
        for i in range(0, len(sp_group_indices), self.batch_size):
            batch = sp_group_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        # Total samples in dataset
        total_samples = len(self.dataset)

        # Account for SP group sharding
        sp_group_samples = total_samples // self.num_sp_groups
        if not self.drop_last and total_samples % self.num_sp_groups != 0:
            sp_group_samples += 1

        # Calculate number of batches
        total_batches = sp_group_samples // self.batch_size
        if not self.drop_last and sp_group_samples % self.batch_size != 0:
            total_batches += 1

        return total_batches


def collate_fn(batch):
    return {
        key: torch.stack([d[key] for d in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [d[key] for d in batch]
        for key in batch[0]
    }


if __name__ == "__main__":
    from accelerate import Accelerator
    from torchdata.stateful_dataloader import StatefulDataLoader

    dataloader_num_workers = 8
    batch_size = 2
    num_train_epochs = 2
    seed = 0

    gan_folder = [
        "/mnt/hdfs/data/ysh_new/userful_things_wan/gan_latents/ultravideo/clips_long_960",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/gan_latents/ultravideo/clips_short_960",
    ]
    ode_folder = [
        "/mnt/hdfs/data/ysh_new/userful_things_wan/ode_pairs/vidprom_filtered_extended",
    ]
    text_folder = [
        "/mnt/hdfs/data/ysh_new/userful_things_wan/text-embedding/mixkit_filter",
        "/mnt/hdfs/data/ysh_new/userful_things_wan/text-embedding/vidprom_filtered_extended",
    ]

    accelerator = Accelerator()
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        gan_folders=gan_folder,
        ode_folders=ode_folder,
        text_folders=text_folder,
        is_use_gt_history=True,
        force_rebuild=True,
        seed=seed,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        seed=seed,
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    dataloader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
    )
    print(len(dataset), len(dataloader))
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

    step = 0
    global_step = 0
    first_epoch = 0
    print("Testing dataloader...")
    dataset_counts = defaultdict(int)
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            # Get metadata
            gan_uttid = batch["gan_uttid"]
            ode_uttid = batch["ode_uttid"]
            text_uttid = batch["text_uttid"]

            # Get feature
            # For GAN
            gan_vae_latents = batch["gan_vae_latents"]
            gan_prompt_raws = batch["gan_prompt_raws"]
            gan_prompt_embeds = batch["gan_prompt_embeds"]
            print(gan_vae_latents.shape, gan_prompt_embeds.shape, gan_prompt_raws)

            # For ODE
            ode_prompt_raws = batch["ode_prompt_raws"]
            ode_prompt_embeds = batch["ode_prompt_embeds"]
            print(ode_prompt_embeds.shape, ode_prompt_raws)

            # For Text
            text_prompt_raws = batch["text_prompt_raws"]
            text_prompt_embeds = batch["text_prompt_embeds"]
            print(text_prompt_embeds.shape, text_prompt_raws)

            if accelerator.process_index == 0:
                # print info
                print(f" Step {step}:")
                print(f"  Batch {i}:")
                print(f"  Batch size: {len(gan_uttid)}")
                print(f"  Uttids: {gan_uttid}, {ode_uttid}, {text_uttid}")
                print(
                    f"  Data Name: {batch['gan_dataset_name']}, {batch['ode_dataset_name']}, {batch['text_dataset_name']}"
                )

            for dataset_name in batch["gan_dataset_name"]:
                dataset_counts[dataset_name] += 1

            step += 1

    print("实际采样统计:", dict(dataset_counts))
