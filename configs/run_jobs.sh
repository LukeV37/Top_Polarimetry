cd ..
for file in configs/configs/*; do
    filename=$(basename "$file" .config)
    echo "File: $file"
    ./run_job.sh $file > configs/logs/$filename.log
    echo ""
done
cd configs
