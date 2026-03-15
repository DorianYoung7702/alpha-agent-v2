"""
OUROBOROS - auto_mine.py
持续因子挖掘脚本，整晚不间断运行
用法：python auto_mine.py
"""
import sys
import time
import subprocess
import smtplib
import socks
import socket
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

PYTHON = r"C:\Users\Administrator\Miniconda3\python.exe"
WORKDIR = Path(r"D:\VC\alpha-agent")
LOG_FILE = WORKDIR / "logs" / "auto_mine.log"
ITER_LOG = WORKDIR / "memory" / "iteration_log.md"
FACTOR_LIB = WORKDIR / "memory" / "factor_library.json"

# 邮件配置
GMAIL = "doriany7702@gmail.com"
GMAIL_PASS = "ierd kfte uxuc whmp"
NOTIFY_TO = "doriany7702@icloud.com"

MAX_ROUNDS = 50       # 最多跑50轮
ROUND_INTERVAL = 30   # 每轮间隔30秒


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def send_email(subject, body):
    try:
        socks.set_default_proxy(socks.HTTP, "127.0.0.1", 7890)
        socket.socket = socks.socksocket
        msg = MIMEMultipart()
        msg["From"] = GMAIL
        msg["To"] = NOTIFY_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL, GMAIL_PASS)
            s.sendmail(GMAIL, NOTIFY_TO, msg.as_string())
        log(f"邮件已发送: {subject}")
    except Exception as e:
        log(f"邮件发送失败: {e}")


def read_iter_log():
    if ITER_LOG.exists():
        return ITER_LOG.read_text(encoding="utf-8")
    return "无迭代记录"


def run_one_round(version="v5b"):
    log(f"���始新一轮挖掘 (version={version})...")
    try:
        result = subprocess.run(
            [PYTHON, "main.py", "--mode", "full", "--version", version],
            cwd=str(WORKDIR),
            timeout=3600,  # 最多1小时
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode == 0:
            log("本轮挖掘完成 ✅")
            return True
        else:
            log(f"本轮挖掘失败 ❌: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        log("超时（1小时），跳过本轮")
        return False
    except Exception as e:
        log(f"异常: {e}")
        return False


def main():
    log("="*50)
    log("OUROBOROS 持续挖掘启动")
    log(f"计划运行 {MAX_ROUNDS} 轮")
    log("="*50)

    send_email(
        "[OUROBOROS] 夜间挖掘启动",
        f"毛毛已启动夜间因子挖掘，计划运行 {MAX_ROUNDS} 轮。\n有结果会通知你。\n\n毛毛 🐾"
    )

    success = 0
    failed = 0

    for i in range(1, MAX_ROUNDS + 1):
        log(f"\n第 {i}/{MAX_ROUNDS} 轮")

        ok = run_one_round(version="v5b")
        if ok:
            success += 1
        else:
            failed += 1

        # 每5轮���一次进度邮件
        if i % 5 == 0:
            iter_summary = read_iter_log()
            send_email(
                f"[OUROBOROS] 第{i}轮进度汇报",
                f"已完成 {i} 轮\n成功: {success} | 失败: {failed}\n\n最新迭代摘要：\n{iter_summary[-1000:]}\n\n毛毛 🐾"
            )

        # 检查是否找到有效因子
        import json
        try:
            lib = json.loads(FACTOR_LIB.read_text(encoding="utf-8"))
            if lib.get("factors"):
                msg = f"找到 {len(lib['factors'])} 个有效因子！\n\n{iter_summary[-500:]}"
                send_email("[OUROBOROS] 🎉 找到有效因子！", msg)
                log("找到有效因子，继续优化...")
        except:
            pass

        if i < MAX_ROUNDS:
            log(f"等待 {ROUND_INTERVAL} 秒后开始下���轮...")
            time.sleep(ROUND_INTERVAL)

    # 最终汇报
    final_summary = read_iter_log()
    send_email(
        "[OUROBOROS] 夜间挖掘完成",
        f"夜间挖掘结束！\n共运行 {MAX_ROUNDS} 轮\n成功: {success} | 失败: {failed}\n\n最终结果：\n{final_summary[-2000:]}\n\n毛毛 🐾"
    )
    log("夜间挖掘全部完成���")


if __name__ == "__main__":
    main()
