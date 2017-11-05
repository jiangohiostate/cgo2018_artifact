all: wcc_serial wcc_mask wcc_invec wcc_grouping 


wcc_serial: wcc_serial.cpp
	icc -xMIC-AVX512  -O2 -no-vec -std=c++11  wcc_serial.cpp -o wcc_serial

wcc_mask: wcc_mask.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 wcc_mask.cpp -o wcc_mask

wcc_invec: wcc_invec.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 wcc_invec.cpp -o wcc_invec

wcc_grouping: wcc_grouping.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 wcc_grouping.cpp -o wcc_grouping

clean: 
	rm wcc_serial wcc_mask wcc_invec wcc_grouping
