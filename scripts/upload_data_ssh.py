"""
将本地仓库的 data/ 目录上传到远程 .../ms-rca-llm/data/（整目录覆盖式传输）。

依赖: pip install paramiko
用法（在仓库根目录，勿把密码写进仓库文件）:
  set SSH_PASSWORD=你的密码
  python scripts/upload_data_ssh.py

或 Linux:
  SSH_PASSWORD='...' python scripts/upload_data_ssh.py

可选环境变量:
  SSH_HOST  默认 connect.bjb2.seetacloud.com
  SSH_PORT  默认 36239
  SSH_USER  默认 root
  REMOTE_DATA_PARENT  默认 /root/autodl-tmp/ms-rca-llm  （data 会传到其下 data/）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_DATA = REPO / "data"


def main() -> None:
    try:
        import paramiko
        from scp import SCPClient
    except ImportError:
        print("请先安装: pip install paramiko scp", file=sys.stderr)
        sys.exit(1)

    password = os.environ.get("SSH_PASSWORD", "").strip()
    if not password:
        print("请设置环境变量 SSH_PASSWORD 后重试（不要在命令行历史里长期保留）", file=sys.stderr)
        sys.exit(1)

    host = os.environ.get("SSH_HOST", "connect.bjb2.seetacloud.com")
    port = int(os.environ.get("SSH_PORT", "36239"))
    user = os.environ.get("SSH_USER", "root")
    remote_parent = os.environ.get("REMOTE_DATA_PARENT", "/root/autodl-tmp/ms-rca-llm").rstrip("/")

    if not LOCAL_DATA.is_dir():
        print(f"本地不存在目录: {LOCAL_DATA}", file=sys.stderr)
        sys.exit(1)

    print(f"本地: {LOCAL_DATA}")
    print(f"远程: {user}@{host}:{port} -> {remote_parent}/data/")
    print("开始上传（体积大时请耐心等待）…")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=user, password=password, timeout=60)

    # 确保远程目录存在
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_parent}/data")
    stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err, file=sys.stderr)

    with SCPClient(ssh.get_transport(), socket_timeout=600) as scp:
        # 将本地 data 目录内容传到远程 .../data/
        for entry in LOCAL_DATA.iterdir():
            remote_path = f"{remote_parent}/data/{entry.name}"
            if entry.is_dir():
                scp.put(str(entry), remote_path=remote_path, recursive=True)
            else:
                scp.put(str(entry), remote_path=remote_path)

    ssh.close()
    print("上传完成。")


if __name__ == "__main__":
    main()
