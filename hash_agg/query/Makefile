all:  bucket_mask bucket_invec linear_serial  linear_invec linear_mask 

linear_serial: linear_serial.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 -no-vec linear_serial.cpp -o linear_serial

linear_simd: linear_simd.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 linear_simd.cpp -o linear_simd

linear_invec: linear_invec.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 linear_invec.cpp -o linear_invec

bucket_invec: bucket_invec.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 bucket_invec.cpp -o bucket_invec

linear_mask: linear_mask.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 linear_mask.cpp -o linear_mask

bucket_mask: bucket_mask.cpp
	icc -D$(SIZE) -O3 -xMIC-AVX512 bucket_mask.cpp -o bucket_mask

clean:
	rm  bucket_mask bucket_invec linear_invec linear_serial linear_mask

