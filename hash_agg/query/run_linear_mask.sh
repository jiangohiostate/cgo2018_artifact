#!/bin/bash

make clean
make SIZE=SMALL
for i in 6 7 8 9 10; do
echo -e "\e[32mcardinality: $i\e[0m"
  ./linear_mask $1\_$i.txt
done

make clean
make SIZE=MEDIUM
for i in 11 12 13 14 15; do
echo -e "\e[32mcardinality: $i\e[0m"
  ./linear_mask $1\_$i.txt
done

make clean
make SIZE=LARGE
for i in 16 17 18 19; do
echo -e "\e[32mcardinality: $i\e[0m"
  ./linear_mask $1\_$i.txt
done
