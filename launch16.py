"""Console/file logging, temporary sleep prevention, graceful Windows interrupt."""
import argparse
import ctypes
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command:
        parser.error('Supply a runner command.')
    args.out.mkdir(parents=True,exist_ok=True)
    windows=sys.platform=='win32'
    flags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0
    sleep_api=None
    if windows:
        sleep_api=ctypes.windll.kernel32.SetThreadExecutionState
        sleep_api.argtypes=[ctypes.c_uint]
        sleep_api.restype=ctypes.c_uint
        if not sleep_api(0x80000001):
            print('Sleep-prevention request unavailable; check VM power policy.',flush=True)
    process=None
    old={}
    try:
        cmd=[sys.executable,'-u',str(Path(__file__).parent/'heat_formal_v3_16/runner.py'),
             '--out',str(args.out.resolve())]+args.command
        process=subprocess.Popen(cmd,creationflags=flags,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        def stop(signum,frame):
            print('Stopping after current updates. Wait for checkpoint saving before closing.',flush=True)
            if process.poll() is None:
                process.send_signal(signal.CTRL_BREAK_EVENT if windows else signal.SIGINT)
        for sig in ([signal.SIGINT,signal.SIGBREAK] if windows else [signal.SIGINT,signal.SIGTERM]):
            old[sig]=signal.signal(sig,stop)
        messages=queue.Queue()
        def reader():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        with (args.out/'terminal.log').open('a',encoding='utf-8',buffering=1) as log:
            while True:
                try:
                    line=messages.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                print(line,end='',flush=True)
                log.write(line)
        process.wait()
        return process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if windows else signal.SIGINT)
            process.wait()
        for sig,handler in old.items():
            signal.signal(sig,handler)
        if sleep_api is not None:
            sleep_api(0x80000000)


if __name__=='__main__':
    sys.exit(main())
