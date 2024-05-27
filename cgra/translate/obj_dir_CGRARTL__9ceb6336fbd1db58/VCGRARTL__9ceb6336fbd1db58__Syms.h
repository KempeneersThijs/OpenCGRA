// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VCGRARTL__9CEB6336FBD1DB58__SYMS_H_
#define _VCGRARTL__9CEB6336FBD1DB58__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VCGRARTL__9ceb6336fbd1db58.h"
#include "VCGRARTL__9ceb6336fbd1db58___024unit.h"

// SYMS CLASS
class VCGRARTL__9ceb6336fbd1db58__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VCGRARTL__9ceb6336fbd1db58*    TOPp;
    
    // CREATORS
    VCGRARTL__9ceb6336fbd1db58__Syms(VCGRARTL__9ceb6336fbd1db58* topp, const char* namep);
    ~VCGRARTL__9ceb6336fbd1db58__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
