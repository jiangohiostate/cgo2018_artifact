#!/bin/bash


options=("hitter" "mcluster" "zipf" "Quit")
PS3='Select an input distribution (1, 2, 3) or Quit (4) and press Enter: '
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
    echo -e "\e[32mversion: linear_serial\e[0m"
    ./run_linear_serial.sh $filename
    echo "=================================="
    echo -e "\e[32mversion: linear_mask\e[0m"
    ./run_linear_mask.sh $filename
    echo "=================================="
    echo -e "\e[32mversion: bucket_mask\e[0m"
    ./run_bucket_mask.sh $filename
    echo "=================================="
    echo -e "\e[32mversion: linear_invec\e[0m"
    ./run_linear_invec.sh $filename
    echo "=================================="
    echo -e "\e[32mversion: bucket_invec\e[0m"
    ./run_bucket_invec.sh $filename
    echo "=================================="
    echo ""
    fi
    echo "1) hitter     3) zipf"
    echo "2) mcluster   4) Quit"
done



