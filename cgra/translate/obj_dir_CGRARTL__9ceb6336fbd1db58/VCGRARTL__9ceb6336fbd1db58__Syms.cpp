// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VCGRARTL__9ceb6336fbd1db58__Syms.h"
#include "VCGRARTL__9ceb6336fbd1db58.h"
#include "VCGRARTL__9ceb6336fbd1db58___024unit.h"



// FUNCTIONS
VCGRARTL__9ceb6336fbd1db58__Syms::VCGRARTL__9ceb6336fbd1db58__Syms(VCGRARTL__9ceb6336fbd1db58* topp, const char* namep)
    // Setup locals
    : __Vm_namep(namep)
    , __Vm_didInit(false)
    // Setup submodule names
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
}
