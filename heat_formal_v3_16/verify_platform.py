"""Native platform smoke checks: locks and process-group interruption, no PDE data."""
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest

import runtime as rt


class PlatformChecks(unittest.TestCase):
    def test_native_exclusive_lock_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            command=[sys.executable,str(Path(__file__).resolve()),'--lock-child',str(path)]
            with rt.lock(path):
                blocked=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(blocked.returncode,3,blocked.stdout+blocked.stderr)
            released=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(released.returncode,0,released.stdout+released.stderr)

    def test_interrupt_api_selects_native_signal(self):
        class Process:
            sent=None
            def poll(self):return None
            def send_signal(self,value):self.sent=value
        process=Process();rt.interrupt_process(process)
        self.assertEqual(process.sent,signal.CTRL_BREAK_EVENT if rt.IS_WINDOWS else signal.SIGINT)

    def test_native_child_process_group_receives_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            proc=rt.spawn([sys.executable,str(Path(__file__).resolve()),'--signal-child',tmp],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+45
                while not (root/'ready').exists():
                    if proc.poll() is not None or time.monotonic()>deadline:
                        self.fail('Child failed to become ready for interrupt test.')
                    time.sleep(.05)
                rt.interrupt_process(proc)
                self.assertEqual(proc.wait(timeout=15),0)
                self.assertTrue((root/'stopped').exists())
            finally:
                if proc.poll() is None:
                    proc.terminate();proc.wait()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--lock-child':
        try:
            with rt.lock(Path(sys.argv[2])):pass
        except RuntimeError:sys.exit(3)
    elif len(sys.argv)>1 and sys.argv[1]=='--signal-child':
        root=Path(sys.argv[2])
        def stop(sig,frame):
            (root/'stopped').write_text('graceful',encoding='utf-8')
            sys.exit(0)
        signal.signal(signal.SIGINT,stop)
        if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,stop)
        (root/'ready').write_text('ready',encoding='utf-8')
        while True:time.sleep(.05)
    else:
        unittest.main(verbosity=2)
