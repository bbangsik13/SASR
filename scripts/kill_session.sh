pkill -TERM -P $(cat run_subshell.pid)
kill $(cat run_subshell.pid)
kill -9 `ps aux | grep python | awk '{print $2}'`
