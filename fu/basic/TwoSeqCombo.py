"""
==========================================================================
TwoSeqComb.py
==========================================================================
Simple generic two sequentially combined functional units for CGRA tile.

Author : Cheng Tan
  Date : November 28, 2019

"""

from pymtl3 import *
from pymtl3.stdlib.ifcs import SendIfcRTL, RecvIfcRTL
from ...lib.opt_type    import *

class TwoSeqCombo( Component ):

  def construct( s, DataType, CtrlType, Fu0, Fu1, num_inports, num_outports,
                 data_mem_size ):

    # Constant
    AddrType = mk_bits( clog2( data_mem_size ) )
    s.const_zero = DataType(0, 0)

    # Interface
    s.recv_in  = [ RecvIfcRTL( DataType ) for _ in range( num_inports  ) ]
    s.recv_const = RecvIfcRTL( DataType )
    s.recv_opt = RecvIfcRTL( CtrlType )
    s.send_out = [ SendIfcRTL( DataType ) for _ in range( num_outports ) ]

    # Redundant interfaces for MemUnit
    s.to_mem_raddr   = SendIfcRTL( AddrType )
    s.from_mem_rdata = RecvIfcRTL( DataType )
    s.to_mem_waddr   = SendIfcRTL( AddrType )
    s.to_mem_wdata   = SendIfcRTL( DataType )

    # Components
    s.Fu0 = Fu0( DataType, CtrlType, 2, 1, data_mem_size )
    s.Fu1 = Fu1( DataType, CtrlType, 2, 1, data_mem_size )

    # Connections
    s.recv_in[0].msg     //= s.Fu0.recv_in[0].msg
    s.recv_in[1].msg     //= s.Fu0.recv_in[1].msg
    s.recv_in[2].msg     //= s.Fu1.recv_in[1].msg

#    s.Fu0.recv_opt.msg //= s.recv_opt.msg
#    s.Fu1.recv_opt.msg //= s.recv_opt.msg

    s.Fu0.send_out[0].msg //= s.Fu1.recv_in[0].msg
    s.Fu1.send_out[0].msg //= s.send_out[0].msg

    s.Fu0.recv_const //= s.recv_const

    @s.update
    def update_signal():
      s.recv_in[0].rdy  = s.send_out[0].rdy
      s.recv_in[1].rdy  = s.send_out[0].rdy
      s.recv_in[2].rdy  = s.send_out[0].rdy
      s.Fu0.recv_opt.en = s.recv_opt.en
      s.Fu1.recv_opt.en = s.recv_opt.en
      s.recv_opt.rdy    = s.send_out[0].rdy
#      s.send_out[0].en  = s.recv_in[0].en and s.recv_in[1].en and\
#                          s.recv_in[2].en and s.recv_opt.en
      s.send_out[0].en  = s.recv_in[0].en and s.recv_opt.en

    @s.update
    def update_mem():
      s.to_mem_waddr.en    = b1( 0 )
      s.to_mem_wdata.en    = b1( 0 )
      s.to_mem_wdata.msg   = s.const_zero
      s.to_mem_waddr.msg   = AddrType( 0 )
      s.to_mem_raddr.msg   = AddrType( 0 )
      s.to_mem_raddr.en    = b1( 0 )
      s.from_mem_rdata.rdy = b1( 0 )

  def line_trace( s ):
    return s.Fu0.line_trace() + " ; " + s.Fu1.line_trace()

