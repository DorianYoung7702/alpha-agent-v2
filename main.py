"""
OUROBOROS - main.py
量化策略挖掘 Agent 入口
用法：python main.py [--mode fetch|run|full] [--version v1-v5] [--no-cache]
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from data.fetcher import fetch_all_symbols, fetch_funding_all, build_panel, SYMBOLS_TOP30

# 日志配置
logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/ouroboros_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day")

Path("logs").mkdir(exist_ok=True)


def cmd_fetch(args):
    logger.info("[Step 1] Fetching market data (Top30)...")
    data = fetch_all_symbols(
        interval="1d",
        start_str="2020-01-01",
        symbols=SYMBOLS_TOP30,
        use_cache=not args.no_cache
    )
    funding = fetch_funding_all(
        start_str="2020-01-01",
        symbols=SYMBOLS_TOP30,
        use_cache=not args.no_cache
    )
    logger.info(f"Fetched {len(data)} symbols, {len(funding)} funding")
    panel = build_panel(data, funding_data=funding, min_history_days=60)
    logger.info(f"Panel shape: {panel.shape}")
    return panel, data


def cmd_run(panel, version="v5", raw_data=None):
    logger.info(f"[Step 2] Starting OUROBOROS loop {version}...")

    if version == "v1":
        from agent.loop import OuroborosLoop
        loop = OuroborosLoop(
            panel=panel, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", top_n=3, max_iterations=100,
            max_no_improve=50, use_market_filter=True
        )
    elif version == "v2":
        from agent.loop_v2 import OuroborosLoopV2
        loop = OuroborosLoopV2(
            panel=panel, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", top_n=3, use_market_filter=True
        )
    elif version == "v3":
        from agent.loop_v3 import OuroborosLoopV3
        loop = OuroborosLoopV3(
            panel=panel, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", top_n=3
        )
    elif version == "v4":
        from agent.loop_v4 import OuroborosLoopV4
        loop = OuroborosLoopV4(
            panel=panel, train_start="2020-01-01", train_end="2023-12-31",
            test_start="2024-01-01", top_n=8, rebalance_days=3,
            use_ic_filter=True, ic_window=30, ic_threshold=0.03,
            use_market_filter=True
        )
    else:  # v5/v5b
        if version == "v5":
            from agent.loop_v5 import OuroborosLoopV5
            loop = OuroborosLoopV5(data=raw_data, train_start="2020-01-01", train_end="2023-12-31", test_start="2024-01-01", oos_sharpe_threshold=0.8)
        else:  # v5b
            from agent.loop_v5b import OuroborosLoopV5b
            loop = OuroborosLoopV5b(data=raw_data, train_start="2020-01-01", train_end="2023-12-31", test_start="2024-01-01", oos_sharpe_threshold=0.8)
    loop.run()


def main():
    parser = argparse.ArgumentParser(description="OUROBOROS Strategy Mining Agent")
    parser.add_argument("--mode", choices=["fetch", "run", "full"], default="full")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch data")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v4", "v5", "v5b"], default="v5b")
    args = parser.parse_args()

    if args.mode == "fetch":
        cmd_fetch(args)
    elif args.mode == "run":
        logger.info("Loading cached data (Top30)...")
        data = fetch_all_symbols(symbols=SYMBOLS_TOP30, use_cache=True)
        funding = fetch_funding_all(symbols=SYMBOLS_TOP30, use_cache=True)
        panel = build_panel(data, funding_data=funding, min_history_days=60)
        cmd_run(panel, version=args.version, raw_data=data)
    else:  # full
        panel, data = cmd_fetch(args)
        if panel is not None and not panel.empty:
            cmd_run(panel, version=args.version, raw_data=data)
        else:
            logger.error("No data available, aborting")
            sys.exit(1)


if __name__ == "__main__":
    main()
