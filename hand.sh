echo "Current shell PID: $$"

while ps -p 113977 113976 > /dev/null 2>&1; do
    echo "$(date): Waiting for processes 113977 and 113976 to finish..."
    sleep 600
done

exec python pretrain.py
