import argparse
import os
import pickle
from pathlib import Path
import uuid
import glob
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoImageProcessor
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from PIL import Image

from model.nnModel import Net


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS_PATH = "data/splits.pkl"
DATA_DIR = "./PetImages"

CLASS_NAMES = {0: "Cat", 1: "Dog"}
LABEL_MAP = {"Cat": 0, "Dog": 1}


# ---------------------------------------------------------------------------
# Splits — store paths + labels, not tensors
# ---------------------------------------------------------------------------

def discover_images(data_dir: str = DATA_DIR) -> tuple[list[str], list[int]]:
	paths, labels = [], []
	for class_name, label in LABEL_MAP.items():
		pattern = os.path.join(data_dir, class_name, "*")
		for p in sorted(glob.glob(pattern)):
			try:
				with Image.open(p) as img:
					img.verify()
				paths.append(p)
				labels.append(label)
			except Exception:
				print(f"  Skipping corrupt file: {p}")
	print("Discovered {} images: {} cats, {} dogs".format(len(paths), labels.count(0), labels.count(1)))
	return paths, labels


def create_splits(seed: int = 42) -> dict:
	paths, labels = discover_images()
	paths = np.array(paths)
	labels = np.array(labels)

	p_train, p_temp, y_train, y_temp = train_test_split(
		paths, labels, test_size=0.2, random_state=seed, stratify=labels
	)
	p_val, p_test, y_val, y_test = train_test_split(
		p_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
	)

	splits = {
		"train": (p_train.tolist(), y_train.tolist()),
		"val": (p_val.tolist(), y_val.tolist()),
		"test": (p_test.tolist(), y_test.tolist()),
	}

	with open(SPLITS_PATH, "wb") as f:
		pickle.dump(splits, f)

	print(f"Splits saved to {SPLITS_PATH}")
	for k, (ps, ys) in splits.items():
		ys_arr = np.array(ys)
		print(f"  {k}: n={len(ps)}  balance(dog)={ys_arr.mean():.2f}")

	return splits


def load_splits() -> dict:
	with open(SPLITS_PATH, "rb") as f:
		return pickle.load(f)


# ---------------------------------------------------------------------------
# Dataset — loads images on the fly with per-model preprocessing
# ---------------------------------------------------------------------------

class CatDogDataset(Dataset):
	def __init__(self, paths: list[str], labels: list[int], transform_fn=None):
		self.paths = paths
		self.labels = labels
		self.transform_fn = transform_fn

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		img = Image.open(self.paths[idx]).convert(self.transform_fn.mode)
		x = self.transform_fn(img)
		y = self.labels[idx]
		return x, y


class CustomCNNTransform:
	"""Grayscale 50x50, scaled to [0,1]."""
	mode = "L"

	def __call__(self, img: Image.Image) -> torch.Tensor:
		img = img.resize((50, 50), Image.BILINEAR)
		arr = np.array(img, dtype=np.float32) / 255.0
		return torch.from_numpy(arr).unsqueeze(0)  # [1, 50, 50]


class ResNetTransform:
	"""RGB 224x224 with ImageNet normalization via HF AutoImageProcessor."""
	mode = "RGB"

	def __init__(self):
		self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

	def __call__(self, img: Image.Image) -> torch.Tensor:
		out = self.processor(images=img, return_tensors="pt")
		return out["pixel_values"].squeeze(0)  # [3, 224, 224]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def prepare_model(model_name: str | None, device: str) -> tuple[nn.Module, float, object]:
	if model_name is None:
		return Net().to(device), 1e-3, CustomCNNTransform()

	if model_name == "resnet18":
		model = AutoModelForImageClassification.from_pretrained(
			"microsoft/resnet-18",
			num_labels=2,
			ignore_mismatched_sizes=True,
		)

		# # freeze backbone, only train classifier head
		# for name, param in model.named_parameters():
		# 	if "classifier" not in name:
		# 		param.requires_grad = False

		return model.to(device), 1e-4, ResNetTransform()

	raise ValueError(f"Unknown model: {model_name}. Supported: None, 'resnet18'")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(split, net, dataloader, device) -> dict:
	print(f"Running evals on split {split}...")
	all_probs, all_preds, all_labels = [], [], []

	net.eval()
	with torch.no_grad():
		for X_batch, y_batch in dataloader:
			X_batch = X_batch.to(device)
			out = net(X_batch)
			logits = out.logits if hasattr(out, "logits") else out
			probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
			preds = logits.argmax(dim=1).cpu().numpy()

			all_probs.append(probs)
			all_preds.append(preds)
			all_labels.append(y_batch.numpy())

	y_true = np.concatenate(all_labels)
	preds = np.concatenate(all_preds)
	probs_pos = np.concatenate(all_probs)

	acc = accuracy_score(y_true, preds)
	prec = precision_score(y_true, preds, zero_division=0)
	rec = recall_score(y_true, preds, zero_division=0)
	f1 = f1_score(y_true, preds, zero_division=0)
	try:
		auc = roc_auc_score(y_true, probs_pos)
	except ValueError:
		auc = float("nan")

	return {
		"accuracy": acc,
		"precision": prec,
		"recall": rec,
		"f1": f1,
		"auc": auc,
		"confusion_matrix": confusion_matrix(y_true, preds),
	}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions_grid(net, dataset, device="cpu", rows=4, cols=5, seed=42):
	n = rows * cols
	rng = np.random.RandomState(seed)
	indices = rng.choice(len(dataset), size=n, replace=False)

	net.eval()
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))

	for i, ax in enumerate(axes.flat):
		x, gt = dataset[indices[i]]
		with torch.no_grad():
			out = net(x.unsqueeze(0).to(device))
			logits = out.logits if hasattr(out, "logits") else out
			pred = logits.argmax(dim=1).item()

		# display: grayscale or undo ImageNet normalization for RGB
		if x.shape[0] == 1:
			ax.imshow(x[0].cpu().numpy(), cmap="gray")
		else:
			img_np = x.cpu().numpy().transpose(1, 2, 0)
			mean = np.array([0.485, 0.456, 0.406])
			std = np.array([0.229, 0.224, 0.225])
			img_np = (img_np * std + mean).clip(0, 1)
			ax.imshow(img_np)

		ax.set_xticks([])
		ax.set_yticks([])
		correct = gt == pred
		ax.set_title(
			f"GT: {CLASS_NAMES[gt]} | Pred: {CLASS_NAMES[pred]}",
			fontsize=10, fontweight="bold",
			color="green" if correct else "red",
		)

	plt.suptitle("Predictions", fontsize=16, fontweight="bold")
	plt.tight_layout()
	return fig


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
	if args.recreate_splits or not os.path.exists(SPLITS_PATH):
		splits = create_splits(seed=args.seed)
	else:
		splits = load_splits()
		print(f"Loaded cached splits from {SPLITS_PATH}")

	device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

	net, default_lr, transform_fn = prepare_model(args.model, device)
	n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

	lr = args.lr if args.lr is not None else default_lr
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
	criterion = nn.CrossEntropyLoss()
	
	print("="*50)
	print(f"Dataset size: train ({len(splits['train'][0])}), val ({len(splits['val'][0])}), test ({len(splits['test'][0])})")
	print(f"Device: {device}")
	print(f"Model: {args.model or 'custom_cnn'}  Trainable params: {n_params:,}")
	print(f"LR = {lr}")
	print("="*50)

	num_workers = min(4, os.cpu_count() or 1)

	train_ds = CatDogDataset(*splits["train"], transform_fn=transform_fn)
	val_ds = CatDogDataset(*splits["val"], transform_fn=transform_fn)
	test_ds = CatDogDataset(*splits["test"], transform_fn=transform_fn)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers) #, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers) #, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers) #, pin_memory=True)

	# train eval subset (same size as val for fair comparison)
	n_eval = len(val_ds)
	rng = np.random.RandomState(args.seed)
	train_eval_idx = rng.choice(len(train_ds), size=min(n_eval, len(train_ds)), replace=False).tolist()
	train_eval_ds = torch.utils.data.Subset(train_ds, train_eval_idx)
	train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers) #, pin_memory=True)

	# --- wandb ---
	if args.log_wandb:
		import wandb
		wandb.init(
			entity="sanjana-learning",
			project="cat-or-dog",
			name=args.run_id,
			config={
				"epochs": args.epochs,
				"batch_size": args.batch_size,
				"lr": lr,
				"model": args.model or "custom_cnn",
				"seed": args.seed,
				"train_size": len(train_ds),
				"val_size": len(val_ds),
				"test_size": len(test_ds),
			},
		)

	ckpt_dir = Path("ckpts") / str(uuid.uuid4())
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	best_val_f1 = -1.0

	for epoch in range(args.epochs):
		print(f"Training epoch {epoch+1}/{args.epochs}...")
		
		net.train()
		running_loss = 0.0
		n_samples = 0

		for bx, by in tqdm(train_loader, desc="Training"):
			bx, by = bx.to(device), by.to(device)

			optimizer.zero_grad(set_to_none=True)
			out = net(bx)
			logits = out.logits if hasattr(out, "logits") else out
			loss = criterion(logits, by)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * bx.size(0)
			n_samples += bx.size(0)

		train_loss = running_loss / n_samples

		# --- evaluate all splits ---
		train_metrics = evaluate("train", net, train_eval_loader, device)
		val_metrics = evaluate("val", net, val_loader, device)
		test_metrics = evaluate("test", net, test_loader, device)

		# --- log ---
		log = {"epoch": epoch + 1, "train/loss": train_loss}
		for split_name, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
			for k, v in metrics.items():
				if k != "confusion_matrix":
					log[f"{split_name}/{k}"] = v

		if args.log_wandb:
			wandb.log(log)

		print(
			f"Epoch {epoch+1}/{args.epochs}  "
			f"loss={train_loss:.4f}  "
			f"train_acc={train_metrics['accuracy']:.3f}  "
			f"val_acc={val_metrics['accuracy']:.3f}  "
			f"test_acc={test_metrics['accuracy']:.3f}"
		)

		torch.save(net.state_dict(), ckpt_dir / "last.pth")
		if val_metrics["f1"] > best_val_f1:
			best_val_f1 = val_metrics["f1"]
			torch.save(net.state_dict(), ckpt_dir / "best.pth")
			print(f"  -> new best val f1: {best_val_f1:.4f}")

	fig = plot_predictions_grid(net, test_ds, device=device)

	if args.log_wandb:
		wandb.log({"test/prediction_grid": wandb.Image(fig)})
		wandb.finish()

	print(f"Checkpoints saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
	p = argparse.ArgumentParser(description="CatDog CNN training")

	p.add_argument("--run-id", type=str, default=None, help="wandb run name")
	p.add_argument("--log-wandb", action="store_true", help="enable wandb logging")

	p.add_argument("--epochs", type=int, default=20)
	p.add_argument("--batch-size", type=int, default=100)

	p.add_argument("--lr", type=float, default=None, help="auto-set per model if omitted")
	p.add_argument("--model", type=str, default=None, choices=[None, "resnet18"])

	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--recreate-splits", action="store_true")
	# p.add_argument("--eval-only", action="store_true")

	p.add_argument("--run-name", type=str, default=None, help="ckpt dir name for eval-only")

	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	
	train(args)