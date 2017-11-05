#include "vint.h"
#include "vfloat.h"
template <typename T>
mask invec_min(mask m, const vint &idx, T& vadd) {
        __m512i vconf = _mm512_maskz_conflict_epi32(m, idx.val);
        mask mconf = _mm512_cmpeq_epi32_mask(vconf, _mm512_set1_epi32(0));
        vadd.val = __invec_min(mconf, idx.val, vadd.val, m);
        return mconf&m;
}

__m512 __invec_min(__mmask16 mconf, __m512i vny, __m512 vdy, __mmask16 mupdate) {

  __mmask16 mconft = mconf;
  __m512 vnewdy = vdy;

  if(mconft != 0xFFFF) { 

    if(!(mconft & 0x2)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+1));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+2));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+3));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x10)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+4));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x20)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+5));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x40)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+6));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x80)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+7));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x100)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+8));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x200)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+9));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x400)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+10));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x800)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+11));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x1000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+12));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x2000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+13));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+14));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+15));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_min_ps(mt, vnewdy)));
      mconft |= mt;
    }
  }
  return vnewdy;
}




template <typename T>
mask invec_add(mask m, const vint &idx, T& vadd) {
        __m512i vconf = _mm512_maskz_conflict_epi32(m, idx.val);
        mask mconf = _mm512_cmpeq_epi32_mask(vconf, _mm512_set1_epi32(0));
        vadd.val = __invec_add(mconf, idx.val, vadd.val, m);
        return mconf&m;
}

__m512 __invec_add(__mmask16 mconf, __m512i vny, __m512 vdy, __mmask16 mupdate) {

  __mmask16 mconft = mconf;
  __m512 vnewdy = vdy;


  if(mconft != 0xFFFF) { 

    if(!(mconft & 0x2)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+1));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+2));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+3));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x10)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+4));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x20)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+5));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x40)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+6));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x80)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+7));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x100)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+8));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x200)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+9));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x400)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+10));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x800)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+11));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x1000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+12));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x2000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+13));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+14));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8000)) {
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+15));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_add_ps(mt, vnewdy)));
      mconft |= mt;
    }
  }
  return vnewdy;
}

