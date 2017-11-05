all: sswp_serial sswp_mask sswp_invec sswp_grouping 


sswp_serial: sswp_serial.cpp
	icc -xMIC-AVX512  -O2 -no-vec -std=c++11  sswp_serial.cpp -o sswp_serial

sswp_mask: sswp_mask.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 sswp_mask.cpp -o sswp_mask

sswp_invec: sswp_invec.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 sswp_invec.cpp -o sswp_invec

sswp_grouping: sswp_grouping.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 sswp_grouping.cpp -o sswp_grouping

clean: 
	rm sswp_serial sswp_mask sswp_invec sswp_grouping
