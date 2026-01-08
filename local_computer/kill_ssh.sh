#!/bin/sh
# Kill SSH local port-forward listeners created for these ports
ports="49100 65432"
killed_any=0

for port in $ports; do
  pids=$(lsof -n -P -iTCP:${port} -sTCP:LISTEN -a -c ssh -t 2>/dev/null | sort -u)
  if [ -n "$pids" ]; then
    echo "Stopping SSH tunnel(s) on localhost:${port} (PID(s): $pids)"
    for pid in $pids; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 0.5
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
    killed_any=1
  else
    echo "No SSH tunnel listening on localhost:${port}"
  fi
done

if [ "$killed_any" -eq 1 ]; then
  echo "SSH tunneling stopped."
else
  echo "No matching SSH tunnels found."
fi