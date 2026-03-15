"""
OUROBOROS - continuous_mine.py
事件驱动持续挖掘：Alpha跑完→Expert分析→Alpha继续
"""
import sys
import time
import json
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
EXPERT_GUIDANCE = WORKDIR / "memory" / "expert_guidance.md"
ITER_LOG = WORKDIR / "memory" / "iteration_log.md"
FACTOR_LIB = WORKDIR / "memory" / "factor_library.json"
OPENCLAW = r"D:\npm-global\openclaw.cmd"

# 邮件
GMAIL = "doriany7702@gmail.com"
GMAIL_PASS = "ierd kfte uxuc whmp"
NOTIFY_TO = "doriany7702@icloud.com"

# 控制
MAX_ROUNDS = 100
WAIT_FOR_EXPERT_SEC = 120  # ���待 Expert 回复最多2分钟
ROUND_INTERVAL = 10


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_path = WORKDIR / "logs" / "continuous_mine.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
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
        log(f"邮件已发: {subject}")
    except Exception as e:
        log(f"邮件失败: {e}")


def notify_expert(round_num, result_summary):
    """通知 Expert 分析结果"""
    try:
        msg = f"[Alpha第{round_num}轮结果]\n{result_summary}\n\n���分析并更新 memory/expert_guidance.md，然后回复 DONE。"
        result = subprocess.run(
            [OPENCLAW, "message", "send",
             "--session", "agent:expert:main",
             "--message", msg],
            capture_output=True, text=True, timeout=30
        )
        log(f"已通知 Expert（轮次{round_num}）")
        return True
    except Exception as e:
        log(f"通知 Expert 失败: {e}")
        return False


def notify_secretary(subject, body):
    """通知产品经理发邮件"""
    try:
        msg = f"[产品经理任务] 请发邮件给 YZX（doriany7702@icloud.com）\n主题：{subject}\n内容：{body}"
        subprocess.run(
            [OPENCLAW, "message", "send",
             "--session", "agent:secretary:main",
             "--message", msg],
            capture_output=True, text=True, timeout=30
        )
        log("已通知产品经理")
    except Exception as e:
        log(f"通知���品经理失败: {e}")


def get_guidance_mtime():
    """获取 expert_guidance.md 的修改时间"""
    if EXPERT_GUIDANCE.exists():
        return EXPERT_GUIDANCE.stat().st_mtime
    return 0


def wait_for_expert_update(before_mtime, timeout=120):
    """等待 Expert 更新指导文件"""
    log(f"等待 Expert 更新（最多{timeout}秒）...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        current_mtime = get_guidance_mtime()
        if current_mtime > before_mtime:
            log("Expert 已更新指导 ✅")
            return True
        time.sleep(5)
    log("Expert 未在时限内更新，使用旧指导继续")
    return False


def run_one_round(version="v5b"):
    """跑一轮回测，返回摘要"""
    log(f"开始回测（{version}）...")
    try:
        result = subprocess.run(
            [PYTHON, "main.py", "--mode", "run", "--version", version],
            cwd=str(WORKDIR),
            timeout=3600,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        # 读取迭代日志摘要
        if ITER_LOG.exists():
            content = ITER_LOG.read_text(encoding="utf-8")
            # 取最后500字
            summary = content[-500:]
        else:
            summary = "无迭代记录"
        return result.returncode == 0, summary
    except subprocess.TimeoutExpired:
        return False, "超时"
    except Exception as e:
        return False, str(e)


def check_good_result():
    """检查是否有好结果（有效因子）"""
    try:
        lib = json.loads(FACTOR_LIB.read_text(encoding="utf-8"))
        return lib.get("factors", [])
    except:
        return []


def main():
    log("=" * 50)
    log("OUROBOROS 事件驱动持续挖掘启动")
    log("=" * 50)

    send_email(
        "[OUROBOROS] 夜间挖掘启动",
        f"毛毛已启动夜间持续挖掘。\n流程：Alpha跑完→Expert分析→Alpha继续\n有好结果立即通知你���\n\n毛毛 🐾"
    )

    success = 0
    for round_num in range(1, MAX_ROUNDS + 1):
        log(f"\n{'='*30}")
        log(f"第 {round_num}/{MAX_ROUNDS} 轮开始")

        # 1. 跑回测
        ok, summary = run_one_round()
        if ok:
            success += 1
            log(f"回测完成 ✅ (累计成功{success}轮)")
        else:
            log(f"回测失败 ❌")

        # 2. 通知 Expert
        before_mtime = get_guidance_mtime()
        notify_expert(round_num, summary)

        # 3. 等 Expert 更新指导
        wait_for_expert_update(before_mtime, timeout=WAIT_FOR_EXPERT_SEC)

        # 4. 检查是否有好结果
        good_factors = check_good_result()
        if good_factors:
            msg = f"找到 {len(good_factors)} 个有效策略！\n\n{summary}"
            log(f"🎉 找到有效策略！")
            # 让产品经理发邮件
            notify_secretary("[OUROBOROS] 找到有效策略！", msg)
            # 同时直接发邮件
            send_email("[OUROBOROS] 🎉 找到有效策略！", msg + "\n\n毛毛 🐾")

        # 5. 每10轮汇报
        if round_num % 10 == 0:
            send_email(
                f"[OUROBOROS] 第{round_num}轮���度",
                f"已完成 {round_num} 轮，成功 {success} 轮。\n\n{summary}\n\n毛毛 🐾"
            )

        time.sleep(ROUND_INTERVAL)

    send_email(
        "[OUROBOROS] 夜间挖掘完成",
        f"共运行 {MAX_ROUNDS} 轮，成功 {success} 轮。\n\n最终结果：\n{ITER_LOG.read_text(encoding='utf-8')[-2000:]}\n\n毛毛 🐾"
    )
    log("全部完成！")


if __name__ == "__main__":
    main()
