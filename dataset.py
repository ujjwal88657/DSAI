"""
data/dataset.py
Hinglish hate speech dataset: loading, simulation, noise injection,
tokenization, and DataLoader construction.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Hinglish hate speech samples (representative, diverse examples)
# ---------------------------------------------------------------------------

HINGLISH_SAMPLES = {
    # label 0 = hate
    0: [
        "Ye log hamare desh ke liye khatarnak hain, inhe bahar nikalo",
        "Is community ko destroy kar do, ye deserve nahi karte",
        "Saale terrorists hain ye sab, police ko pakadna chahiye",
        "Inhe maaro, ye desh ke dushman hain",
        "Koi inse baat mat karo, gande log hain ye",
        "Is religion ke log kabhi achhe nahi ho sakte",
        "Unhe vote mat do, traitors hain ye sab",
        "Ye log jhooth bolte hain, inpe trust mat karo",
        "Inhe desh se bhaga do, hamare liye nahi hain ye",
        "Saale crooks hain, sab corrupt hain",
        "Is group ko band kar do, society ke liye dangerous hain",
        "Unka koi future nahi, bas pareshaan karte hain",
        "Ye log apni aukat mein rahen, hamare barabar nahi",
        "Nahi chahiye aisi community yahan pe",
        "Inpe action lena chahiye government ko, bahut ho gaya",
        "Ye sab milke conspiracy kar rahe hain hamare khilaf ",
        "Inhe citizenship nahi deni chahiye, foreigners hain",
        "Kuch nahi hoga inse, waste of time hai baat karna",
        "Is religion wale log kabhi peaceful nahi hote",
        "Ye log desh todna chahte hain, arrest karo inhe",
    ],
    # label 1 = offensive
    1: [
        "Bhai tu pagal hai kya, aisa kaise kar sakta hai",
        "Yaar ye toh complete bakwaas hai jo tune kaha",
        "Are chup kar, kuch nahi pata tujhe is baare mein",
        "Idiot hai kya, dekh ke nahi chalta",
        "Teri toh aadat hi kharab hai, kuch nahi seekha tune",
        "Besharam kahin ka, sharam nahi aati",
        "Tu toh ekdum dumb hai yaar seriously",
        "Stupid log hain ye, kuch nahi pata inhe",
        "Chutiya chhap film hai ye, time waste mat karo",
        "Gawar hai bhai tu, city mein aake bhi nahi seekha kuch",
        "Yaar ye toh cheapest kaam kar diya tune",
        "Kya loser hai bhai, kuch kaam nahi usse",
        "Bekar hai ye sab, faltu time waste hai",
        "Bakwaas band kar apni, boring ho gaye ho",
        "Tu samjhta kya hai apne aap ko madharchod, nothing special",
        "Yaar itna attitude kyun hai tujhe, aukaat dekh apni",
        "Kuch nahi ho sakta tere se, waste hai",
        "Frustrated insaan hai ye, ignore karo isse",
        "Kya ullu ki tarah behave kar raha hai",
        "Bhai ye toh full cringe hai, embarrassing hai yaar",
    ],
    # label 2 = neutral
    2: [
        "Aaj ka match bahut accha tha, India ne achi performance ki",
        "Naya movie dekha, storyline interesting thi",
        "Yaar kya plan hai weekend ke liye, kuch karte hain",
        "Bhai khaana bahut tasty tha aaj restaurant mein",
        "Office mein meeting thi, project update diya",
        "Aaj mausam bahut accha hai, thodi walk ki",
        "Nayi book padh raha hoon, bahut informative hai",
        "Birthday party thi dost ki, bahut maza aaya",
        "Online shopping ki, delivery kal tak aayegi",
        "Gym gaya aaj, workout bahut tiring thi",
        "Series dekh raha hoon, plot twist bahut interesting hai",
        "Bhai coffee peete hain kuch baat karte hain",
        "Exam ki preparation kar raha hoon, stress ho raha hai",
        "Travel plan bana rahe hain, Goa trip ke liye",
        "Nayi recipe try ki, bahut accha bana",
        "Cricket match dekhte hain aaj, India vs Pak hai",
        "College life bahut busy ho gayi hai in dino",
        "Freelance project mila hai, interesting hai kaam",
        "Music sun raha hoon, stress kum hota hai",
        "Family ke sath dinner kiya, bahut achha time tha",
        "Tech news padh raha tha, AI bahut advance ho raha hai",
        "Coding kar raha hoon side project ke liye",
        "Photography hobby hai meri, nature shots lete hain",
        "Koi achhi podcast recommend karo na",
        "Bhai kya scene hai, kab milte hain yaar",
    ],
}


def generate_hinglish_dataset(
    n_samples: int = 3000,
    class_distribution: Optional[List[float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Hinglish hate speech dataset by sampling
    from template sentences and adding slight variations.
    """
    random.seed(seed)
    np.random.seed(seed)

    if class_distribution is None:
        class_distribution = [0.25, 0.35, 0.40]   # hate / offensive / neutral

    class_counts = [int(n_samples * p) for p in class_distribution]
    class_counts[-1] = n_samples - sum(class_counts[:-1])  # fix rounding

    records = []
    variations = [
        " yaar", " bhai", " dost", " seriously", " actually",
        " honestly", " matlab", " like", " toh", " na",
        ""  # no suffix
    ]

    for label, count in enumerate(class_counts):
        templates = HINGLISH_SAMPLES[label]
        for _ in range(count):
            base = random.choice(templates)
            variation = random.choice(variations)
            text = base + variation
            # Add random noise characters occasionally
            if random.random() < 0.05:
                text = text + " " + random.choice(["!!!", "???", "...", "😤", "🤔"])
            records.append({"text": text, "label": label, "original_label": label})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def inject_symmetric_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip each label to a uniformly random OTHER class with probability noise_rate."""
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    noise_mask = rng.rand(len(labels)) < noise_rate
    for i in np.where(noise_mask)[0]:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = rng.choice(choices)
    return noisy, noise_mask


def inject_asymmetric_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip each label to the NEXT class (cyclically) with probability noise_rate.
    Mimics real-world confusable-class noise (hate↔offensive, offensive↔neutral).
    """
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    noise_mask = rng.rand(len(labels)) < noise_rate
    flip_map = {i: (i + 1) % num_classes for i in range(num_classes)}
    for i in np.where(noise_mask)[0]:
        noisy[i] = flip_map[labels[i]]
    return noisy, noise_mask


def inject_instance_noise(
    labels: np.ndarray,
    features: Optional[np.ndarray],
    noise_rate: float,
    num_classes: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instance-dependent noise: higher noise probability for 'harder' examples
    (approximated by random feature-based difficulty scores since we don't
    have embeddings at data-loading time).
    """
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    # Simulate difficulty as random scores (will be replaced by real features later)
    difficulty = rng.rand(len(labels))
    flip_probs = noise_rate * difficulty / difficulty.mean()
    flip_probs = np.clip(flip_probs, 0, 0.9)
    noise_mask = rng.rand(len(labels)) < flip_probs
    for i in np.where(noise_mask)[0]:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = rng.choice(choices)
    return noisy, noise_mask


def apply_noise(
    df: pd.DataFrame,
    noise_type: str,
    noise_rate: float,
    num_classes: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply chosen noise type to the dataframe's labels."""
    labels = df["label"].values.astype(int)

    if noise_type == "symmetric":
        noisy_labels, mask = inject_symmetric_noise(labels, noise_rate, num_classes, seed)
    elif noise_type == "asymmetric":
        noisy_labels, mask = inject_asymmetric_noise(labels, noise_rate, num_classes, seed)
    elif noise_type == "instance":
        noisy_labels, mask = inject_instance_noise(labels, None, noise_rate, num_classes, seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    df = df.copy()
    df["original_label"] = labels
    df["label"] = noisy_labels
    df["is_noisy"] = mask.astype(int)
    actual_noise = mask.mean()
    print(f"[Noise] type={noise_type}, requested={noise_rate:.2%}, actual={actual_noise:.2%}")
    return df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HinglishDataset(Dataset):
    """
    PyTorch Dataset for Hinglish hate speech.
    Handles tokenization and returns per-sample indices for
    noise-tracking methods.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_len: int = 128,
        indices: Optional[List[int]] = None,
        original_labels: Optional[List[int]] = None,
        is_noisy: Optional[List[int]] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.indices = indices if indices is not None else list(range(len(texts)))
        self.original_labels = original_labels if original_labels is not None else labels
        self.is_noisy = is_noisy if is_noisy is not None else [0] * len(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        orig_label = int(self.original_labels[idx])
        noisy_flag = int(self.is_noisy[idx])
        sample_idx = int(self.indices[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "original_label": torch.tensor(orig_label, dtype=torch.long),
            "is_noisy": torch.tensor(noisy_flag, dtype=torch.long),
            "index": torch.tensor(sample_idx, dtype=torch.long),
            "text": text,
        }

    def update_labels(self, new_labels: np.ndarray):
        """Used by label correction methods to refurbish labels in-place."""
        self.labels = new_labels.tolist()


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class DataModule:
    """
    Manages dataset creation, noise injection, splitting, and DataLoader
    construction. Acts as the single entry point for all data operations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dcfg = cfg.data
        self.tcfg = cfg.training
        self.tokenizer = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def _resolve_column(self, df: pd.DataFrame, preferred: str, role: str) -> str:
        """Resolve a configured column name, allowing case-insensitive matches."""
        if preferred in df.columns:
            return preferred

        lower_to_actual = {c.lower(): c for c in df.columns}
        if preferred.lower() in lower_to_actual:
            return lower_to_actual[preferred.lower()]

        raise ValueError(
            f"Configured {role} column '{preferred}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    def _load_source_dataframe(self) -> pd.DataFrame:
        """Load the configured CSV. Fall back to synthetic data only if it is absent."""
        dataset_path = getattr(self.dcfg, "dataset_path", None)
        if dataset_path:
            dataset_path = os.path.abspath(dataset_path)
            if os.path.exists(dataset_path):
                print(f"[DataModule] Loading full dataset from {dataset_path}")
                return pd.read_csv(dataset_path)
            print(f"[DataModule] Dataset not found at {dataset_path}; using synthetic fallback.")

        print("[DataModule] Generating synthetic Hinglish dataset...")
        return generate_hinglish_dataset(n_samples=3000, seed=self.dcfg.random_seed)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize source data to text/label/original_label/is_noisy columns and
        infer class count from the full CSV.
        """
        text_col = self._resolve_column(df, self.dcfg.text_column, "text")
        label_col = self._resolve_column(df, self.dcfg.label_column, "label")

        before = len(df)
        work = df.dropna(subset=[text_col, label_col]).copy()
        work["text"] = (
            work[text_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        work = work[work["text"].str.len() > 0].copy()

        numeric_labels = pd.to_numeric(work[label_col], errors="coerce")
        if numeric_labels.notna().all():
            raw_labels = numeric_labels.astype(int)
            unique_raw = sorted(raw_labels.unique().tolist())
            label_map = {old: new for new, old in enumerate(unique_raw)}
            work["label"] = raw_labels.map(label_map).astype(int)
        else:
            raw_labels = work[label_col].astype(str)
            unique_raw = sorted(raw_labels.unique().tolist())
            label_map = {old: new for new, old in enumerate(unique_raw)}
            work["label"] = raw_labels.map(label_map).astype(int)

        work["original_label"] = work["label"].astype(int)
        work["is_noisy"] = 0

        num_classes = int(work["label"].nunique())
        if num_classes < 2:
            raise ValueError("Dataset must contain at least two classes.")

        self.dcfg.num_classes = num_classes
        self.cfg.model.num_classes = num_classes
        if len(self.dcfg.class_names) != num_classes:
            if label_col.lower() == "hate_label" and num_classes == 2:
                self.dcfg.class_names = ["non_hate", "hate"]
            else:
                self.dcfg.class_names = [f"class_{i}" for i in range(num_classes)]

        dropped = before - len(work)
        print(
            f"[DataModule] Prepared {len(work):,} rows "
            f"({dropped:,} dropped as empty/missing)."
        )
        print(f"[DataModule] Label mapping: {label_map}")
        print(f"[DataModule] Classes: {self.dcfg.class_names}")
        return work.reset_index(drop=True)

    def _stratified_split(self, df: pd.DataFrame, test_size: float):
        """Use stratification when each class has enough examples."""
        counts = df["label"].value_counts()
        stratify = df["label"] if len(counts) > 1 and counts.min() >= 2 else None
        return train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=self.dcfg.random_seed,
        )

    def setup(self):
        """Full data pipeline: load CSV -> split -> noise on train -> tokenize."""
        print("\n[DataModule] Setting up data pipeline...")

        # 1. Load and normalize the complete configured dataset.
        df = self._normalize_dataframe(self._load_source_dataframe())

        # 2. Split on clean labels.
        train_val_df, test_df = self._stratified_split(df, self.dcfg.test_ratio)
        val_size_adjusted = self.dcfg.val_ratio / (1 - self.dcfg.test_ratio)
        train_df, val_df = self._stratified_split(train_val_df, val_size_adjusted)

        # 3. Inject optional synthetic noise only into the training labels.
        train_df = train_df.copy()
        if self.dcfg.simulate_noise and self.dcfg.noise_rate > 0:
            train_df = apply_noise(
                train_df,
                noise_type=self.dcfg.noise_type,
                noise_rate=self.dcfg.noise_rate,
                num_classes=self.dcfg.num_classes,
                seed=self.dcfg.noise_seed,
            )
        else:
            train_df["original_label"] = train_df["label"].astype(int)
            train_df["is_noisy"] = 0

        # Validation and test stay clean for fair evaluation.
        val_df = val_df.copy()
        test_df = test_df.copy()
        val_df["label"] = val_df["original_label"].astype(int)
        test_df["label"] = test_df["original_label"].astype(int)
        val_df["is_noisy"] = 0
        test_df["is_noisy"] = 0

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        # Save a processed copy for inspection, but never use it to replace the source CSV.
        os.makedirs(self.dcfg.processed_dir, exist_ok=True)
        cache_path = os.path.join(self.dcfg.processed_dir, f"{self.dcfg.dataset_name}_processed.csv")
        df.to_csv(cache_path, index=False)

        print(f"[DataModule] Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        self._print_label_distribution()

        # 4. Tokenizer
        print(f"[DataModule] Loading tokenizer: {self.dcfg.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.dcfg.tokenizer_name)

    def _print_label_distribution(self):
        for split_name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            dist = df["label"].value_counts().sort_index()
            print(f"  {split_name} label dist: {dict(dist)}")
        if "is_noisy" in self.train_df.columns:
            nr = self.train_df["is_noisy"].mean()
            print(f"  Train noise rate: {nr:.2%}")

    def _make_dataset(self, df: pd.DataFrame, shuffle_labels: bool = False) -> HinglishDataset:
        labels = df["label"].values
        if shuffle_labels:
            labels = np.random.permutation(labels)
        orig_labels = df.get("original_label", df["label"]).values
        is_noisy = df.get("is_noisy", pd.Series([0] * len(df))).values
        return HinglishDataset(
            texts=df["text"].tolist(),
            labels=labels.tolist(),
            tokenizer=self.tokenizer,
            max_len=self.dcfg.max_seq_len,
            indices=df.index.tolist(),
            original_labels=orig_labels.tolist(),
            is_noisy=is_noisy.tolist(),
        )

    def get_train_dataset(self) -> HinglishDataset:
        return self._make_dataset(self.train_df)

    def get_val_dataset(self) -> HinglishDataset:
        return self._make_dataset(self.val_df)

    def get_test_dataset(self) -> HinglishDataset:
        return self._make_dataset(self.test_df)

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        ds = self.get_train_dataset()
        return DataLoader(
            ds,
            batch_size=self.tcfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    def get_val_loader(self) -> DataLoader:
        ds = self.get_val_dataset()
        return DataLoader(ds, batch_size=self.tcfg.batch_size * 2, shuffle=False, num_workers=0)

    def get_test_loader(self) -> DataLoader:
        ds = self.get_test_dataset()
        return DataLoader(ds, batch_size=self.tcfg.batch_size * 2, shuffle=False, num_workers=0)

    def get_paired_train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return two independent shuffled DataLoaders for Co-Teaching."""
        ds1 = self.get_train_dataset()
        ds2 = self.get_train_dataset()
        loader1 = DataLoader(ds1, batch_size=self.tcfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
        loader2 = DataLoader(ds2, batch_size=self.tcfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
        return loader1, loader2
