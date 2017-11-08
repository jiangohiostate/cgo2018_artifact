
echo -e "\e[33m The input files are pulled from dropbox, please make sure you have access to dropbox... \e[0m"
echo "Getting data for graph applications .... "
cd graphs
wget https://www.dropbox.com/s/1guu18eba6hr5ot/datasets.tar?dl=0 -O datasets.tar
tar xf datasets.tar
rm datasets.tar

echo "Getting data for Moldyn .... "
cd ../moldyn
wget https://www.dropbox.com/s/gthaz7lzoz94wno/input.tar?dl=0 -O input.tar
tar xf input.tar
rm input.tar

echo "Generating data for hash-based aggregation ..... "
echo -e "\e[33mIt will take about one hour to generate all the data... \e[0m"
cd ../hash_agg
cd datasets
for i in 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
echo "heavy hitter: group cardinality "$i
python gen_hitter.py $i > hitter_$i\.txt
echo "moving cluster: group cardinality "$i
python gen_mcluster.py $i > mcluster_$i\.txt
echo "zipf: group cardinality "$i
python gen_zipf.py $i > zipf_$i\.txt
done
