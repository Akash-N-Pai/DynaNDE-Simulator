#!/bin/bash
echo "=== Current System Memory ==="
free -h | head -2
echo ""
echo "=== Docker Container Limits ==="
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    echo "Memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes | numfmt --to=iec)"
    echo "Current usage: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes | numfmt --to=iec)"
elif [ -f /sys/fs/cgroup/memory.max ]; then
    echo "Memory limit: $(cat /sys/fs/cgroup/memory.max)"
    echo "Current usage: $(cat /sys/fs/cgroup/memory.current | numfmt --to=iec)"
else
    echo "Cannot detect cgroup memory limits (might be unlimited)"
fi
echo ""
echo "=== OOM Killer Status ==="
dmesg | grep -i "out of memory" | tail -3
