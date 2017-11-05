all: sssp_serial sssp_mask sssp_invec sssp_grouping 


sssp_serial: sssp_serial.cpp
	icc -xMIC-AVX512  -O2 -no-vec -std=c++11  sssp_serial.cpp -o sssp_serial

sssp_mask: sssp_mask.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 sssp_mask.cpp -o sssp_mask

sssp_invec: sssp_invec.cpp
	icc -xMIC-AVX512  -O2 -DMIC -I../../SSE_API_Package/SSE_Template/ -std=c++11 sssp_invec.cpp -o sssp_invec

sssp_grouping: sssp_grouping.cpp
	icc -xMIC-AVX512  -O2 -std=c++11 sssp_grouping.cpp -o sssp_grouping

clean: 
	rm sssp_serial sssp_mask sssp_invec sssp_grouping
