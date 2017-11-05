all: page_rank_serial page_rank_serial_notiling page_rank_grouping page_rank_invec page_rank_mask 

page_rank_serial_notiling:page_rank_serial_notiling.cpp
	icc -xMIC_AVX512 -std=c++11 -O3 -no-vec -o page_rank_serial_notiling page_rank_serial_notiling.cpp -w

page_rank_serial:page_rank_serial.cpp
	icc -xMIC_AVX512 -std=c++11 -O3 -no-vec -o page_rank_serial page_rank_serial.cpp -w

page_rank_invec:page_rank_invec.cpp
	icc -xMIC_AVX512 -std=c++11 -O3 -DMIC -I../../SSE_API_Package/SSE_Template/ -o page_rank_invec page_rank_invec.cpp -w 

page_rank_mask:page_rank_mask.cpp
	icc -xMIC_AVX512 -std=c++11 -O3 -o page_rank_mask page_rank_mask.cpp -w 

page_rank_grouping:page_rank_grouping.cpp
	icc -xMIC_AVX512 -std=c++11 -O3 -o page_rank_grouping page_rank_grouping.cpp -w 

clean:
	rm page_rank_serial_notiling page_rank_serial page_rank_grouping page_rank_invec page_rank_mask
