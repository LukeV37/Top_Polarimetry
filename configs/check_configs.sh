for file in configs/*; do
    echo "File: $file"
    sed -n '14p;24p;25p' "$file"
    echo ""
done
