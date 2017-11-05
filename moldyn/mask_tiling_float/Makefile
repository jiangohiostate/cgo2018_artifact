PROG = moldyn
MISC = misc
TIME = timer

LOCAL_CFLAGS = -DCSTYLE 
INC = 
DEFS = -DTIME -DLREORDER -DCOMPUTE_SORT -DCONVERT -DORIG_ADAPTIVE -DTILESIZE=32684
CFLAGS = -O2 $(DEFS) $(LOCAL_CFLAGS) $(INC) -xMIC_AVX512
CPPFLAGS = -O2 $(DEFS) $(LOCAL_CFLAGS) $(INC) -std=c++11 -xMIC_AVX512
LDFLAGS = -O2 -xMIC_AVX512
CC = gcc 
CCC = icc 

LIBS = -lm

OBJS=moldyn.o \
         misc.o\
	 timer.o

main: $(OBJS)
	$(CCC) $(LDFLAGS) -o main $(OBJS)
%.o : %.cpp
	$(CCC) $(CPPFLAGS) -c $< -o $*.o -xMIC_AVX512
%.o : %.c
	$(CCC) $(CFLAGS) -c $< -o $*.o $(LIBS) -xMIC_AVX512

clean:
	-rm main $(OBJS)

#-- Sequential running -----------------------------
#org: $(PROG).c $(TIME).c
#	$(CC) $(CFLAGS) -DBLOCK_ADPATIVE -o $(PROG) $(PROG).c $(MISC).c $(TIME).c $(LIBS)
#
#org0: $(PROG).c $(TIME).c
#	$(CC) $(CFLAGS) -DORIG_ADAPTIVE -o $(PROG)0 $(PROG).c $(MISC).c $(TIME).c $(LIBS)
