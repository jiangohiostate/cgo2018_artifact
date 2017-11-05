make

echo "Moldyn Tiling_and_Mask"
echo "input: 16-3.0r"
./main < moldyn.in16.30
echo "---------------------"
echo "input: 32-3.0r"
./main < moldyn.in32.30
echo "======================"
