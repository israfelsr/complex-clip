#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class AroWrap:
    def __init__(self, model, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = None
        if device is None:
            device = model.device
        self.device = device

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i : min(num_text, i + text_batch_size)]
            text_feats = self.model.encode_text(text, self.device)
            if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            image_feats = self.model.encode_image(batch, self.device)
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

    @torch.no_grad()
    def process_batch(self, b):
        width = len(b["caption_options"])
        bs = len(b["caption_options"][0])

        all_entries = []
        for cap_tuple in b["caption_options"]:
            all_entries += list(cap_tuple)

        entries_tokenized = self.model.tokenizer(
            all_entries, return_tensors="pt", padding=True
        ).to(self.device)
        pixel_values = b["image_options"][0]["pixel_values"][0]
        all_logits = self.model.model(
            input_ids=entries_tokenized["input_ids"],
            attention_mask=entries_tokenized["attention_mask"],
            pixel_values=pixel_values.to(self.device),
        )

        def do_keep(a):
            rowsize = width * bs

            curr_row = a // rowsize
            curr_col = a % bs
            return curr_col == curr_row

        index_np = np.arange(width * bs * bs).reshape((bs, width * bs))
        grouped = all_logits.logits_per_image.cpu().numpy()[do_keep(index_np)]

        scores = grouped.reshape((bs, 1, width))
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            batch_score = self.process_batch(batch)
            scores.append(batch_score)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores


class AROtoHFCLIPWrap:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            device = model.device
        self.device = device

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i : min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(
                text, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            text_feats = self.model.get_text_features(**text_input)
            if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            images = batch["image"]["pixel_values"][0].to(self.device)
            image_feats = self.model.get_image_features(images)
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

    @torch.no_grad()
    def process_batch(self, b):
        width = len(b["caption_options"])
        bs = len(b["caption_options"][0])

        all_entries = []
        for cap_tuple in b["caption_options"]:
            all_entries += list(cap_tuple)
        entries_tokenized = self.tokenizer(
            all_entries, return_tensors="pt", padding=True
        ).to(self.device)
        pixel_values = b["image_options"][0]["pixel_values"][0]
        all_logits = self.model(
            input_ids=entries_tokenized["input_ids"],
            attention_mask=entries_tokenized["attention_mask"],
            pixel_values=pixel_values.to(self.device),
        )

        def do_keep(a):
            rowsize = width * bs

            curr_row = a // rowsize
            curr_col = a % bs
            return curr_col == curr_row

        index_np = np.arange(width * bs * bs).reshape((bs, width * bs))
        grouped = all_logits.logits_per_image.cpu().numpy()[do_keep(index_np)]

        scores = grouped.reshape((bs, 1, width))
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            batch_score = self.process_batch(batch)
            scores.append(batch_score)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores


class AROtoOpenCLIPWrap:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            device = model.device
        self.device = device

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i : min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(text, truncate=True).to(self.device)
            text_feats = self.model.encode_text(text_input)
            if normalize:
                text_feats = F.normalize(text_feats, dim=-1)
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:

            images = batch["image"].to(self.device)
            image_feats = self.model.encode_image(images)
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

    @torch.no_grad()
    def process_batch(self, b):
        width = len(b["caption_options"])
        bs = len(b["caption_options"][0])

        all_entries = []
        for cap_tuple in b["caption_options"]:
            all_entries += list(cap_tuple)
        entries_tokenized = self.tokenizer(
            all_entries, return_tensors="pt", padding=True
        ).to(self.device)
        pixel_values = b["image_options"][0]["pixel_values"][0]
        all_logits = self.model(
            input_ids=entries_tokenized["input_ids"],
            attention_mask=entries_tokenized["attention_mask"],
            pixel_values=pixel_values.to(self.device),
        )

        def do_keep(a):
            rowsize = width * bs

            curr_row = a // rowsize
            curr_col = a % bs
            return curr_col == curr_row

        index_np = np.arange(width * bs * bs).reshape((bs, width * bs))
        grouped = all_logits.logits_per_image.cpu().numpy()[do_keep(index_np)]

        scores = grouped.reshape((bs, 1, width))
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            batch_score = self.process_batch(batch)
            scores.append(batch_score)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores
