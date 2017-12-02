#!/bin/bash

make 

options=("higgs-twitter.txt" "amazon0312.txt" "soc-pokec.txt" "Quit")
PS3='Select an input file (1, 2, 3) or Quit (4) and press Enter: '
select name in "${options[@]}" 
do
if [ ! -z $name ]
then
    if [ $name == 'Quit' ]
    then
        break
    fi
    filename=../datasets/$name
    echo -e "\e[32minput: $filename\e[0m"
    echo -e "\e[32mversion: nontiling_serial\e[0m"
    ./wcc_serial $filename
    echo "=================================="
    echo -e "\e[32mversion: tiling_and_grouping\e[0m"
    echo -e "\e[33mgrouping-and-tiling time is not printed out here since the procedure is the same as in pagerank\e[0m"
    ./wcc_grouping $filename
    echo "=================================="
    echo -e "\e[32mversion: nontiling_and_mask\e[0m"
    ./wcc_mask $filename
    echo "=================================="
    echo -e "\e[32mversion: nontiling_and_invec (our approach)\e[0m"
    ./wcc_invec $filename
    echo "=================================="
    echo ""
    fi
    echo "1) higgs-twitter.txt  3) soc-pokec.txt"
    echo "2) amazon0312.txt     4) Quit"
done



