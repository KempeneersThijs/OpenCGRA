// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VCGRARTL__9ceb6336fbd1db58.h for the primary calling header

#ifndef _VCGRARTL__9CEB6336FBD1DB58___024UNIT_H_
#define _VCGRARTL__9CEB6336FBD1DB58___024UNIT_H_  // guard

#include "verilated.h"

//==========

class VCGRARTL__9ceb6336fbd1db58__Syms;

//----------

VL_MODULE(VCGRARTL__9ceb6336fbd1db58___024unit) {
  public:
    
    // INTERNAL VARIABLES
  private:
    VCGRARTL__9ceb6336fbd1db58__Syms* __VlSymsp;  // Symbol table
  public:
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VCGRARTL__9ceb6336fbd1db58___024unit);  ///< Copying not allowed
  public:
    VCGRARTL__9ceb6336fbd1db58___024unit(const char* name = "TOP");
    ~VCGRARTL__9ceb6336fbd1db58___024unit();
    
    // INTERNAL METHODS
    void __Vconfigure(VCGRARTL__9ceb6336fbd1db58__Syms* symsp, bool first);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
