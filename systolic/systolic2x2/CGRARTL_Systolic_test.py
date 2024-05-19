"""
==========================================================================
CGRACL_fir_demo_test.py
==========================================================================
Test cases for CGRAs with CL data/config memory.

Author : Cheng Tan
  Date : Dec 28, 2019

"""

from pymtl3                       import *
from pymtl3.stdlib.test           import TestSinkCL
from pymtl3.stdlib.test.test_srcs import TestSrcRTL

from ...lib.opt_type              import *
from ...lib.messages              import *
from ...lib.ctrl_helper           import *

from ...fu.flexible.FlexibleFuRTL import FlexibleFuRTL
from ...fu.single.AdderRTL        import AdderRTL
from ...fu.single.MulRTL          import MulRTL
from ...fu.single.LogicRTL        import LogicRTL
from ...fu.single.CompRTL         import CompRTL
from ...fu.single.BranchRTL       import BranchRTL
from ...fu.single.PhiRTL          import PhiRTL
from ...fu.single.ShifterRTL      import ShifterRTL
from ...fu.single.MemUnitRTL      import MemUnitRTL
from ..CGRARTL                    import CGRARTL
from ...fu.double.SeqMulAdderRTL  import SeqMulAdderRTL

from ...lib.dfg_helper            import *

import os

#-------------------------------------------------------------------------
# Test harness
#-------------------------------------------------------------------------

class TestHarness( Component ):

  def construct( s, DUT, FunctionUnit, FuList, DataType, PredicateType,
                 CtrlType, width, height, ctrl_mem_size, data_mem_size,
                 src_opt, ctrl_waddr, preload_data, preload_const, psums ):

    s.num_tiles  = width * height
    AddrType     = mk_bits( clog2( ctrl_mem_size ) )

    s.src_opt    = [ TestSrcRTL( CtrlType, src_opt[i] )
                     for i in range( s.num_tiles ) ]
    s.ctrl_waddr = [ TestSrcRTL( AddrType, ctrl_waddr[i] )
                     for i in range( s.num_tiles ) ]
    s.sink_out   = [ TestSinkCL( DataType, psums[i] )
                     for i in range( height-1 ) ]

    s.dut        = DUT( DataType, PredicateType, CtrlType, width, height,
                        ctrl_mem_size, data_mem_size, 100, FunctionUnit, FuList,
                        preload_data, preload_const )
    s.DataType   = DataType

    for i in range( s.num_tiles ):
      connect( s.src_opt[i].send,     s.dut.recv_wopt[i]  )
      connect( s.ctrl_waddr[i].send,  s.dut.recv_waddr[i] )
    
    for i in range( height-1 ):
      connect( s.dut.send_data[i],  s.sink_out[i].recv )
  
  def line_trace( s ):
    return s.dut.line_trace()



def run_sim( test_harness, max_cycles=12 ):
  test_harness.elaborate()
  test_harness.apply( SimulationPass() )
  test_harness.sim_reset()

  # Run simulation
  ncycles = 0
  print()
  print( "{}:{}".format( ncycles, test_harness.line_trace() ))
  while ncycles < max_cycles:
    test_harness.tick()
    ncycles += 1
    print( "{}:{}".format( ncycles, test_harness.line_trace() ))

  test_harness.tick()
  test_harness.tick()
  test_harness.tick()

  print( '=' * 70 )
  print( "----------------- RTL test ------------------" )
  print("=================================================================")
  print("···························· split ······························")
  print("=================================================================")

def test_Systolic2x2():
  target_json = "systolic2x2.json"
  script_dir  = os.path.dirname(__file__)
  file_path   = os.path.join( script_dir, target_json )

  II                = 5
  num_tile_inports  = 4
  num_tile_outports = 4
  num_xbar_inports  = 6
  num_xbar_outports = 8
  ctrl_mem_size     = 8
  width             = 2
  height            = 3
  num_tiles         = width * height
  RouteType         = mk_bits( clog2( num_xbar_inports + 1 ) )
  AddrType          = mk_bits( clog2( ctrl_mem_size ) )
  num_tiles         = width * height
  ctrl_mem_size     = II
  data_mem_size     = 8
  num_fu_in         = 3
  DUT               = CGRARTL
  FunctionUnit      = FlexibleFuRTL
  targetFuList      = [AdderRTL, MemUnitRTL, SeqMulAdderRTL]
  DataType          = mk_data( 16, 1 )
  PredicateType     = mk_predicate( 1, 1 )
  CtrlType          = mk_ctrl( num_fu_in, num_xbar_inports, num_xbar_outports )
  FuInType          = mk_bits( clog2( num_fu_in + 1 ) )
  pickRegister      = [ FuInType( 0 ) for x in range( num_fu_in ) ]

  cgra_ctrl         = CGRACtrl( file_path, CtrlType, RouteType, width, height,
                                num_fu_in, num_xbar_inports, num_xbar_outports,
                                II )
  src_opt           = cgra_ctrl.get_ctrl()

  ctrl_waddr = [ [ AddrType( i ) for i in range( II ) ] for _ in range( num_tiles ) ]

  preload_data   = [DataType(1, 1), DataType(2, 1), DataType(3, 1), DataType(4, 1), DataType(2, 1), DataType(8, 1), DataType(5, 1), DataType(9, 1)] #inputs
  preload_const = [[DataType(0, 1), 
                    DataType(1, 1), 
                    DataType(0, 0), 
                    DataType(0, 0),
                    DataType(0, 0),
                    DataType(4, 1),
                    DataType(5, 1)],
                   [DataType(0, 0), 
                    DataType(2, 1), 
                    DataType(3, 1),
                    DataType(0, 0),
                    DataType(0, 0),
                    DataType(0, 0),
                    DataType(6, 1),
                    DataType(7, 1)], # offset address used for loading
                   [DataType(2, 1)]*II + [DataType(7, 1)]*II, 
                   [DataType(4, 1)]*II + [DataType(9, 1)]*II,
                   [DataType(6, 1)]*II + [DataType(6, 1)]*II,
                   [DataType(8, 1)]*II + [DataType(8, 1)]*II] # preloaded weights
  """
  1 3 2 5     2 6    psums = [14, 20, 59, 137], [30, 44, 52, 120] 
  2 4 8 9     4 8   
              7 6
              9 8
  """
  psums = [[DataType(14, 1), DataType(20, 1), DataType(59, 1), DataType(137, 1)], [DataType(30, 1), DataType(44, 1), DataType(52, 1), DataType(120, 1)]]

  th = TestHarness( DUT, FunctionUnit, targetFuList, DataType, PredicateType,
                    CtrlType, width, height, ctrl_mem_size, data_mem_size,
                    src_opt, ctrl_waddr, preload_data, preload_const, psums )

  for i in range( num_tiles ):
    th.set_param("top.dut.tile["+str(i)+"].construct", FuList=targetFuList)

  run_sim( th )

#def test_CGRA():
#
#  # Attribute of CGRA
#  width     = 4
#  height    = 4
#  DUT       = CGRA
#  FuList    = [ ALU ]
#  DataType  = mk_data( 16 )
#  CtrlType  = mk_ctrl()
#  cgra_ctrl = CGRACtrl( "control_signal.json", CtrlType )
#
#  # FL golden reference
#  fu_dfg    = DFG( "dfg.json" )
#  data_mem  = acc_fl( fu_dfg, DataType, CtrlType )
#
#  th = TestHarness( DUT, FuList, cgra_ctrl, DataType, CtrlType,
#                    width, height, data_mem ) 
#  run_sim( th )

