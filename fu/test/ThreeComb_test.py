"""
==========================================================================
ThreeComb_test.py
==========================================================================
Test cases for two parallelly combined functional unit followed by single
functional unit.

Author : Cheng Tan
  Date : November 29, 2019

"""

from pymtl3 import *
from pymtl3.stdlib.test           import TestSinkCL
from pymtl3.stdlib.test.test_srcs import TestSrcRTL

from ..Alu                import Alu
from ..Shifter            import Shifter
from ..Mul                import Mul
from ..ThreeMulAluShifter import ThreeMulAluShifter
from ..opt_type           import *

#-------------------------------------------------------------------------
# Test harness
#-------------------------------------------------------------------------

class TestHarness( Component ):

  def construct( s, FunctionUnit, DataType,
                 src0_msgs, src1_msgs, src2_msgs, src3_msgs,
                 config_msgs0, config_msgs1, config_msgs2,
                 sink_msgs ):

    s.src_in0   = TestSrcRTL( DataType, src0_msgs    )
    s.src_in1   = TestSrcRTL( DataType, src1_msgs    )
    s.src_in2   = TestSrcRTL( DataType, src2_msgs    )
    s.src_in3   = TestSrcRTL( DataType, src3_msgs    )
    s.src_opt0  = TestSrcRTL( DataType, config_msgs0 )
    s.src_opt1  = TestSrcRTL( DataType, config_msgs1 )
    s.src_opt2  = TestSrcRTL( DataType, config_msgs2 )
    s.sink_out  = TestSinkCL( DataType, sink_msgs    )

    s.dut = FunctionUnit( DataType )

    connect( s.src_in0.send,  s.dut.recv_in0  )
    connect( s.src_in1.send,  s.dut.recv_in1  )
    connect( s.src_in2.send,  s.dut.recv_in2  )
    connect( s.src_in3.send,  s.dut.recv_in3  )
    connect( s.src_opt0.send, s.dut.recv_opt0 )
    connect( s.src_opt1.send, s.dut.recv_opt1 )
    connect( s.src_opt2.send, s.dut.recv_opt2 )
    connect( s.dut.send_out,  s.sink_out.recv )

  def done( s ):
    return s.src_in0.done()  and s.src_in1.done()  and\
           s.src_in2.done()  and s.src_in3.done()  and\
           s.src_opt0.done() and s.src_opt1.done() and\
           s.src_opt2.done() and s.sink_out.done()

  def line_trace( s ):
    return s.dut.line_trace()

def run_sim( test_harness, max_cycles=1000 ):
  test_harness.elaborate()
  test_harness.apply( SimulationPass )
  test_harness.sim_reset()

  # Run simulation

  ncycles = 0
  print()
  print( "{}:{}".format( ncycles, test_harness.line_trace() ))
  while not test_harness.done() and ncycles < max_cycles:
    test_harness.tick()
    ncycles += 1
    print( "{}:{}".format( ncycles, test_harness.line_trace() ))

  # Check timeout

  assert ncycles < max_cycles

  test_harness.tick()
  test_harness.tick()
  test_harness.tick()

def test_mul_alu_shifter():
  FU = ThreeMulAluShifter
  DataType = Bits16
  src_in0  = [ DataType(1), DataType(2),  DataType(4) ]
  src_in1  = [ DataType(2), DataType(3),  DataType(3) ]
  src_in2  = [ DataType(1), DataType(3),  DataType(3) ]
  src_in3  = [ DataType(1), DataType(2),  DataType(2) ]
  sink_out = [ DataType(8), DataType(12), DataType(6) ]
  src_opt0 = [ DataType(OPT_MUL), DataType(OPT_MUL), DataType(OPT_MUL) ]
  src_opt1 = [ DataType(OPT_ADD), DataType(OPT_SUB), DataType(OPT_SUB) ]
  src_opt2 = [ DataType(OPT_LLS), DataType(OPT_LLS), DataType(OPT_LRS) ]
  th = TestHarness( FU, DataType, src_in0, src_in1, src_in2, src_in3,
                    src_opt0, src_opt1, src_opt2, sink_out )
  run_sim( th )

