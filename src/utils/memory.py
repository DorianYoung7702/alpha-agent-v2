"""
Memory Utils - 因子库和迭代日志读写
"""
import json
import datetime
from pathlib import Path
from loguru import logger

MEMORY_DIR = Path(__file__).parent.parent.parent / "memory"
FACTOR_LIBRARY = MEMORY_DIR / "factor_library.json"
FAILED_FACTORS = MEMORY_DIR / "failed_factors.json"
ITERATION_LOG = MEMORY_DIR / "iteration_log.md"


def load_factor_library() -> dict:
    with open(FACTOR_LIBRARY, "r", encoding="utf-8") as f:
        return json.load(f)


def save_factor_library(data: dict):
    with open(FACTOR_LIBRARY, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_failed_factors() -> dict:
    with open(FAILED_FACTORS, "r", encoding="utf-8") as f:
        return json.load(f)


def save_failed_factors(data: dict):
    with open(FAILED_FACTORS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_success(metrics: dict, factor_class: str, params: dict):
    """记录成功因子到因子库"""
    library = load_factor_library()
    entry = {
        "factor_class": factor_class,
        "params": params,
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat()
    }
    library["factors"].append(entry)
    save_factor_library(library)
    logger.info(f"[Memory] Saved success: {metrics['factor_name']}")


def record_failure(factor_name: str, reason: str, metrics: dict, params: dict):
    """记录失败因子，避免重复尝试"""
    failed = load_failed_factors()
    entry = {
        "factor_name": factor_name,
        "reason": reason,
        "metrics": metrics,
        "params": params,
        "timestamp": datetime.datetime.now().isoformat()
    }
    failed["factors"].append(entry)
    save_failed_factors(failed)
    logger.info(f"[Memory] Saved failure: {factor_name} - {reason}")


def is_already_tried(factor_name: str) -> bool:
    """检查因子是否已尝试过（成功或失败）"""
    failed = load_failed_factors()
    failed_names = {f["factor_name"] for f in failed["factors"]}
    library = load_factor_library()
    success_names = {f["metrics"]["factor_name"] for f in library["factors"]}
    return factor_name in failed_names or factor_name in success_names


def append_iteration_log(iteration: int, summary: str):
    """追加迭代日志"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ITERATION_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## 迭代 {iteration} [{timestamp}]\n")
        f.write(summary)
        f.write("\n")


def update_iteration_header(iteration: int, success_count: int, fail_count: int):
    """更新迭代日志头部状态"""
    content = ITERATION_LOG.read_text(encoding="utf-8")
    # 替换状态行
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("- 当前迭代："):
            new_lines.append(f"- 当前迭代：{iteration}")
        elif line.startswith("- 有效因子数："):
            new_lines.append(f"- 有效因子数：{success_count}")
        elif line.startswith("- 失败因子数："):
            new_lines.append(f"- 失败因子数：{fail_count}")
        else:
            new_lines.append(line)
    ITERATION_LOG.write_text("\n".join(new_lines), encoding="utf-8")
