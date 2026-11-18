"""
Конфигурация эксперимента: загрузка YAML, проверка значений, развёртывание сетки.

Базовый конфиг (configs/base.yaml) задаёт все параметры. Конфиг эксперимента
(lr_sweep.yaml, grid_*.yaml) ссылается на базовый через `base:` и меняет только:
  overrides            — поля, фиксированные в этом эксперименте (ключи через точку);
  grid.jobs_over       — значения, перебор которых даёт отдельные задания (процессы);
  grid.replicas_over   — значения, перебор которых даёт реплики внутри задания;
  replicas.seeds       — seed реплик.

Одно задание обучает R реплик в одном батче: у реплик могут различаться только
seed и rho (см. docs/ARCHITECTURE.md §4).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

# Допустимые секции и ключи базового конфига
SCHEMA: dict[str, set[str]] = {
    "experiment": {"name", "output_root"},
    "data": {"q", "train_frac", "split_seed"},
    "model": {"d_hidden", "n_hidden_layers", "activation"},
    "optimizer": {"name", "lr", "betas", "eps"},
    "regularization": {"kind", "rho", "apply_to"},
    "training": {"epochs", "dtype"},
    "eval": {"every", "checkpoint_steps"},
    "replicas": {"seeds"},
    "runtime": {"device", "deterministic", "threads"},
}

# Поля, которые могут различаться у реплик одного задания: Adam хранит одну
# скорость обучения на тензор параметров, поэтому lr сюда не входит
REPLICA_FIELDS = {"regularization.rho"}

EXPERIMENT_KEYS = {"base", "experiment", "overrides", "grid", "replicas"}


@dataclass(frozen=True)
class DataConfig:
    q: int
    train_frac: float
    split_seed: int


@dataclass(frozen=True)
class ModelConfig:
    d_hidden: int
    n_hidden_layers: int
    activation: str


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    eps: float


@dataclass(frozen=True)
class RegularizationConfig:
    kind: str
    rho: tuple[float, ...]  # по одному значению на реплику
    apply_to: str


@dataclass(frozen=True)
class Config:
    """Полностью развёрнутый конфиг одного задания (R реплик)."""

    experiment_name: str
    output_root: str
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    regularization: RegularizationConfig
    epochs: int
    dtype: str
    eval_every: int
    checkpoint_steps: tuple[int, ...]
    seeds: tuple[int, ...]  # по одному значению на реплику
    device: str
    deterministic: bool
    threads: int

    @property
    def n_replicas(self) -> int:
        return len(self.seeds)

    def to_dict(self) -> dict[str, Any]:
        """Словарь из встроенных типов Python: для YAML, JSON и хеша."""
        return json.loads(json.dumps(asdict(self)))


def is_prime(n: int) -> bool:
    """Проверка простоты перебором делителей до sqrt(n)."""
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return raw


def _check_base_schema(raw: dict[str, Any], path: Path) -> None:
    """Неизвестная секция или ключ — ошибка: опечатка не должна молча стать значением по умолчанию."""
    for section, keys in raw.items():
        if section not in SCHEMA:
            raise ValueError(f"{path}: неизвестная секция '{section}'")
        if not isinstance(keys, dict):
            raise ValueError(f"{path}: секция '{section}' должна быть словарём")
        unknown = set(keys) - SCHEMA[section]
        if unknown:
            raise ValueError(f"{path}: неизвестные ключи в '{section}': {sorted(unknown)}")
        missing = SCHEMA[section] - set(keys)
        if missing:
            raise ValueError(f"{path}: в '{section}' не хватает ключей: {sorted(missing)}")
    missing_sections = set(SCHEMA) - set(raw)
    if missing_sections:
        raise ValueError(f"{path}: не хватает секций: {sorted(missing_sections)}")


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    """Записать значение по ключу вида 'section.key' с проверкой по схеме."""
    parts = dotted.split(".")
    if len(parts) != 2 or parts[0] not in SCHEMA or parts[1] not in SCHEMA[parts[0]]:
        raise ValueError(f"неизвестное поле '{dotted}'")
    tree[parts[0]][parts[1]] = value


def _build(tree: dict[str, Any], rhos: list[float], seeds: list[int]) -> Config:
    """Собрать Config из дерева базового конфига и списков rho и seed по репликам."""
    cfg = Config(
        experiment_name=str(tree["experiment"]["name"]),
        output_root=str(tree["experiment"]["output_root"]),
        data=DataConfig(
            q=int(tree["data"]["q"]),
            train_frac=float(tree["data"]["train_frac"]),
            split_seed=int(tree["data"]["split_seed"]),
        ),
        model=ModelConfig(
            d_hidden=int(tree["model"]["d_hidden"]),
            n_hidden_layers=int(tree["model"]["n_hidden_layers"]),
            activation=str(tree["model"]["activation"]),
        ),
        optimizer=OptimizerConfig(
            name=str(tree["optimizer"]["name"]),
            lr=float(tree["optimizer"]["lr"]),
            betas=tuple(float(b) for b in tree["optimizer"]["betas"]),
            eps=float(tree["optimizer"]["eps"]),
        ),
        regularization=RegularizationConfig(
            kind=str(tree["regularization"]["kind"]),
            rho=tuple(float(r) for r in rhos),
            apply_to=str(tree["regularization"]["apply_to"]),
        ),
        epochs=int(tree["training"]["epochs"]),
        dtype=str(tree["training"]["dtype"]),
        eval_every=int(tree["eval"]["every"]),
        checkpoint_steps=tuple(int(t) for t in (tree["eval"]["checkpoint_steps"] or [])),
        seeds=tuple(int(s) for s in seeds),
        device=str(tree["runtime"]["device"]),
        deterministic=bool(tree["runtime"]["deterministic"]),
        threads=int(tree["runtime"]["threads"]),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Проверка типов и диапазонов; при нарушении — ValueError с понятным текстом."""
    errors = []
    if not is_prime(cfg.data.q):
        errors.append(f"q = {cfg.data.q} должно быть простым")
    if not 0 < cfg.data.train_frac < 1:
        errors.append("train_frac должна быть в (0, 1)")
    if cfg.model.d_hidden < 1 or cfg.model.n_hidden_layers < 1:
        errors.append("d_hidden и n_hidden_layers должны быть >= 1")
    if cfg.model.activation != "relu":
        errors.append("поддерживается только activation: relu")
    if cfg.optimizer.name not in {"adam", "sgd"}:
        errors.append("optimizer.name: adam или sgd")
    if cfg.optimizer.lr <= 0:
        errors.append("lr должна быть > 0")
    if len(cfg.optimizer.betas) != 2 or not all(0 <= b < 1 for b in cfg.optimizer.betas):
        errors.append("betas: два числа в [0, 1)")
    if cfg.regularization.kind not in {"none", "l1", "l2"}:
        errors.append("regularization.kind: none, l1 или l2")
    if any(r < 0 for r in cfg.regularization.rho):
        errors.append("rho должно быть >= 0")
    if cfg.regularization.kind == "none" and any(r != 0 for r in cfg.regularization.rho):
        errors.append("при kind: none все rho должны быть 0")
    if cfg.regularization.apply_to != "weights":
        errors.append("поддерживается только apply_to: weights (смещения не штрафуются)")
    if cfg.epochs < 1 or cfg.eval_every < 1:
        errors.append("epochs и eval.every должны быть >= 1")
    if cfg.dtype not in {"float32", "float64"}:
        errors.append("dtype: float32 или float64")
    if any(not 0 <= t <= cfg.epochs for t in cfg.checkpoint_steps):
        errors.append("checkpoint_steps должны лежать в [0, epochs]")
    if not cfg.seeds or len(cfg.seeds) != len(cfg.regularization.rho):
        errors.append("на каждую реплику нужны seed и rho")
    if cfg.device not in {"auto", "cpu", "cuda"}:
        errors.append("device: auto, cpu или cuda")
    if cfg.threads < 1:
        errors.append("threads должно быть >= 1")
    if errors:
        raise ValueError("ошибки в конфиге: " + "; ".join(errors))


def load_experiment(path: str | Path) -> list[Config]:
    """Развернуть конфиг в список заданий.

    Базовый конфиг даёт одно задание: реплики = seeds, rho у всех одно.
    Конфиг эксперимента даёт по заданию на каждую комбинацию grid.jobs_over,
    а внутри задания реплики = комбинации grid.replicas_over x seeds.
    """
    path = Path(path)
    raw = _read_yaml(path)

    if "base" not in raw:
        _check_base_schema(raw, path)
        seeds = list(raw["replicas"]["seeds"])
        return [_build(raw, [raw["regularization"]["rho"]] * len(seeds), seeds)]

    unknown = set(raw) - EXPERIMENT_KEYS
    if unknown:
        raise ValueError(f"{path}: неизвестные секции эксперимента: {sorted(unknown)}")
    base_path = (path.parent / raw["base"]).resolve()
    tree = _read_yaml(base_path)
    _check_base_schema(tree, base_path)
    tree = copy.deepcopy(tree)

    for key, value in (raw.get("experiment") or {}).items():
        _set_dotted(tree, f"experiment.{key}", value)
    for dotted, value in (raw.get("overrides") or {}).items():
        _set_dotted(tree, dotted, value)
    for key, value in (raw.get("replicas") or {}).items():
        _set_dotted(tree, f"replicas.{key}", value)

    grid = raw.get("grid") or {}
    unknown_grid = set(grid) - {"jobs_over", "replicas_over"}
    if unknown_grid:
        raise ValueError(f"{path}: неизвестные ключи grid: {sorted(unknown_grid)}")
    jobs_over = grid.get("jobs_over") or {}
    replicas_over = grid.get("replicas_over") or {}
    for dotted in replicas_over:
        if dotted not in REPLICA_FIELDS:
            raise ValueError(f"{path}: поле '{dotted}' не может различаться у реплик одного задания")

    seeds = list(tree["replicas"]["seeds"])
    rho_values = list(replicas_over.get("regularization.rho", [tree["regularization"]["rho"]]))
    replica_rho = [rho for rho in rho_values for _ in seeds]
    replica_seed = [seed for _ in rho_values for seed in seeds]

    job_keys = list(jobs_over)
    configs = []
    for combo in itertools.product(*(jobs_over[k] for k in job_keys)):
        job_tree = copy.deepcopy(tree)
        for dotted, value in zip(job_keys, combo):
            _set_dotted(job_tree, dotted, value)
        configs.append(_build(job_tree, replica_rho, replica_seed))
    return configs


def load_config(path: str | Path) -> Config:
    """Конфиг ровно одного задания (для scripts/train.py)."""
    configs = load_experiment(path)
    if len(configs) != 1:
        raise ValueError(f"{path} описывает {len(configs)} заданий; используйте scripts/run_grid.py")
    return configs[0]


def config_hash(cfg: Config) -> str:
    """Короткий sha1 канонического JSON конфига (без каталога результатов)."""
    payload = cfg.to_dict()
    payload.pop("output_root")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def job_id(cfg: Config) -> str:
    """Читаемый идентификатор задания: тип регуляризации, lr и хеш конфига."""
    return f"{cfg.regularization.kind}-lr{cfg.optimizer.lr:g}-{config_hash(cfg)}"
