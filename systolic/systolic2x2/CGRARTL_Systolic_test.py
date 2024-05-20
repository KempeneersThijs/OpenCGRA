from pymtl3                       import *
from pymtl3.stdlib.test.test_sinks import TestSinkRTL
from pymtl3.stdlib.test.test_srcs import TestSrcRTL
import numpy as np

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

from ..systolic_workload_helper   import *
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
    s.sink_out   = [ TestSinkRTL(DataType, psums[i]) for i in range(height-1) ] #check correctness
    s.dut        = DUT( DataType, PredicateType, CtrlType, width, height,
                        ctrl_mem_size, data_mem_size, 100, FunctionUnit, FuList,
                        preload_data, preload_const )
    s.DataType   = DataType

    for i in range( s.num_tiles ):
      connect( s.src_opt[i].send,     s.dut.recv_wopt[i]  )
      connect( s.ctrl_waddr[i].send,  s.dut.recv_waddr[i] )
    
    for i in range( height-1 ):
      connect( s.dut.send_data[i],  s.sink_out[i].recv ) #connect tile 3 and 5 EAST port (see CGRARTL.py for connections) to the sink_out
  
  def line_trace( s ):
    return s.dut.line_trace()



def run_sim( test_harness, max_cycles=15 ):
  safety_cycles = max_cycles + 2 #2 is a safety margin for start signal
  test_harness.elaborate()
  test_harness.apply( SimulationPass() )
  test_harness.sim_reset()

  # Run simulation
  ncycles = 0
  print()
  print( "{}:{}".format( ncycles, test_harness.line_trace() ))
  while ncycles < safety_cycles:
    test_harness.tick()
    ncycles += 1
    print( "{}:{}".format( ncycles, test_harness.line_trace() ))
  
  print( '=' * 120 )
  print("execution latency: ", max_cycles, "cycles\n")



def test_Systolic2x2():
  target_json = "systolic2x2.json"
  script_dir  = os.path.dirname(__file__)
  file_path   = os.path.join( script_dir, target_json )

  II                = 5 # 1 extra cycle due to memory loading of the bottom tiles
  num_tile_inports  = 4
  num_tile_outports = 4
  num_xbar_inports  = 6
  num_xbar_outports = 8
  ctrl_mem_size     = 8
  width             = 2
  height            = 3 # 2 bottom tiles used for loading elements from memory
  num_tiles         = width * height
  RouteType         = mk_bits( clog2( num_xbar_inports + 1 ) )
  AddrType          = mk_bits( clog2( ctrl_mem_size ) )
  num_tiles         = width * height
  ctrl_mem_size     = II
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
  
  """
  equation: O[b][k] += I[b][c] * W[c][k]
  Limits of this implementation:
    - It is always assumed the entire array is used (no mapping to limit the used array size) <-> ZigZag
    - the array has to be a square, with the addition of tiles along 1 of the dimensions, used for loading. This makes
      it so the height of the array is always the width + 1. The actual MAC array is a square.
    - b, c and k have to be multiples of the array width (in this case 2). This means that if you imagine a tile
      of the array size (2x2 in this case), this tile has to be able to fit a whole number of times inside both
      the input matrix and the weight matrix.
    - The bottom loops of the temporal mapping are always:
              for b0 from 0 .. [arraywidth]
                parfor c0 from 0 .. [arraywidth]
                parfor k0 from 0 .. [arraywidth]
      - These bottom loops are executed in (arraywidth + (arraywidth+arraywidth-2))+1 cycles (== II)
      - explanation:  if it were a multiplier array that could process the parfor loops in 1 cycle, it would only take
                    arraywidth cycles. Because it is a systolic array, the cycles that it takes to run it in a systolic
                    array fashion have to be added, which is (arraywidth+arraywidth-2). The 1 extra added cycle at the
                    end comes from the extra loading tiles at the bottom of the array. Because of this, one extra
                    loading cycle has to be taken into account.
      - This makes it a relatively naïve implementation, but to improve this, means to manually configure the config.json
        file custom to every workload. A possible improvement to this implementation is a way to automate the generation
        of the config.json file, dependent on every workload seperately.
      - The total amount of cycles is then equal to the remaining temporal loops multiplied by this II.
  """
  #define workload
  I = np.array([
      [1, 3, 2, 5], #4, 3, 2, 1],
      [2, 4, 8, 9], #2, 5, 3, 4],
      [2, 5, 3, 4], #2, 4, 8, 9],
      [4, 3, 2, 1], #, 1, 3, 2, 5],
      [4, 3, 2, 1], #, 1, 3, 2, 5],
      [2, 5, 3, 4], #2, 4, 8, 9],
      [2, 4, 8, 9], #2, 5, 3, 4],
      [1, 3, 2, 5], #4, 3, 2, 1]
      ])

  W = np.array([
      [2, 6, 7, 3, 9, 8, 6, 9],
      [4, 8, 2, 1, 7, 6, 3, 5],
      [7, 6, 3, 5, 4, 8, 2, 1],
      [9, 8, 6, 9, 2, 6, 7, 3],
#      [9, 8, 6, 9, 2, 6, 7, 3],
#      [7, 6, 3, 5, 4, 8, 2, 1],
#      [4, 8, 2, 1, 7, 6, 3, 5],
#      [2, 6, 7, 3, 9, 8, 6, 9]
      ])
  
  preload_const, preload_data, psums, cycles, data_mem_size = get_workload( DataType, I, W, width, II )

  th = TestHarness( DUT, FunctionUnit, targetFuList, DataType, PredicateType,
                    CtrlType, width, height, ctrl_mem_size, data_mem_size,
                    src_opt, ctrl_waddr, preload_data, preload_const, psums )

  for i in range( num_tiles ):
    th.set_param("top.dut.tile["+str(i)+"].construct", FuList=targetFuList)

  run_sim( th, max_cycles=cycles) 

