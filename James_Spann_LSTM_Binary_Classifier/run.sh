echo "Checking if files are in the right place..."
if [ -e modelIndexFileTest.json ]
then
    echo "modelIndexFileTest.json - ACCEPTED"
else
    echo "modelIndexFileTest.json - NOT IN THE DIRECTORY"
    exit 0
fi

if [ -e modelIndexFile.json ]
then
    echo "modelIndexFile.json - ACCEPTED"
else
    echo "modelIndexFile.json - NOT IN THE DIRECTORY"
    exit 0
fi

echo "Warning: Might break due to memory issues. I suggest you close all other applications."

python3 main.py trainterm
