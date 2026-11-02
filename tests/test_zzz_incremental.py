"""临时验证用：验证 _status_plugin 边测边更新（跑完后删除本文件）。"""
import time


def test_slow_then_fail():
    time.sleep(6)
    assert False, "intentional fail after sleep"
