
`timescale 1 ns / 1 ps

	module NPU_v1_0
#(
	// Parameters of NPU Logic
    parameter integer OPCODE_WIDTH	= 4,
    parameter integer BUFFER_ID_WIDTH = 2,
    parameter integer KERNEL_SIZE = 2,
    parameter integer STRIDE = 2,
    parameter integer NUM_DATA = 16,
    parameter integer DRAM_ADDR_WIDTH = 32,
    parameter integer DATA_WIDTH = 8,
    parameter integer BLOCK_WIDTH = 6,
    parameter integer BLOCK_HEIGHT = 16,
    parameter integer ARRAY_BITCOUNT = 5, // ARRAY_WIDTH = 2**ARRAY_BITCOUNT
    parameter integer KE_NUM = 18, // Max Kernel elements num = (7x7)49 x 2048 + 1(redundant)
    parameter integer ADDR_WIDTH = 16, 
    // This value is for Instruction, BRAM ADDR_WIDTH is 15 for Zynq
    // BRAM ADDR_WIDTH : 14:SoC, Each buffer size : 0.5MB, Total Buffer size: 1.5MB(2 Data buffer, 1 weight buffer) // 15: Zynq(1MBx3=3MB), 18: Versal
    parameter integer IMAGE_WH = 10,
    parameter integer NUM_GROUP = 10,
    parameter integer OUTCH_OFFSET = 10,
    parameter integer OUTBLOCK_WIDTH = 6,
    parameter integer IMAGE_COUNT = 12,
    parameter integer ARRAY_WIDTH = 32,
    parameter integer REG_WIDTH = 32,
    parameter integer BUF_WIDTH = 256,
    parameter integer ARRAY_OUTPUT_WIDTH = 1024,
    parameter integer OTHER_DATA_WIDTH=256,
    parameter integer AXI_LITE_DATA_WIDTH=32,
    parameter integer AXI_STREAM_DATA_WIDTH=256,
    parameter integer A_BIT=256,
    parameter integer B_BIT=256,
    parameter integer C_BIT=256,
    parameter integer Scale=32,//float 32
    parameter integer ZP=8,
    
    parameter integer C_S00_AXI_DATA_WIDTH	= 32,
	parameter integer C_S00_AXI_ADDR_WIDTH	= 6,
    parameter integer C_S_AXIS_TDATA_WIDTH	= 256,
    parameter integer C_M_AXIS_TDATA_WIDTH	= 256,
    parameter integer C_M_AXIS_START_COUNT	= 32,
    parameter integer C_AXI_LITE_WIDTH = 32
)
(
/*
// for Testbench without AXI Lite submodule (old version) **************************************************** 
    input wire mem_read_start, mem_write_start, mem_read_op, mem_write_op,
	input wire [BUFFER_ID_WIDTH-1:0] dma_read_buffer_id, dma_write_buffer_id,
	input wire [ADDR_WIDTH-1:0] dma_read_base_addr, dma_write_base_addr,
	input wire [NUM_DATA-1:0] dma_read_num_data, dma_write_num_data,
	output mem_read_done, mem_write_done, 
// MAC-Conv3d, Conv1x1
    output wire mac_op_done,
    input wire mac_start, conv3d, conv1x1, depthconv, matmul, mac_bias_onoff, mac_norm_onoff,
    input wire [OPCODE_WIDTH-1:0] process_ops,
    input wire [BUFFER_ID_WIDTH-1:0] mac_read_db_id,
    input wire [BUFFER_ID_WIDTH-1:0] mac_write_db_id,
    input wire [OUTCH_OFFSET-1:0] mac_store_outch_offset,
    input wire [ADDR_WIDTH-1:0] mac_wb_base_addr,
    input wire [ADDR_WIDTH-1:0] mac_db_base_addr,
    input wire [ADDR_WIDTH-1:0] mac_store_base_addr,
    input wire [ADDR_WIDTH-1:0] mac_bias_base_addr,
    input wire [ADDR_WIDTH-1:0] mac_wzero_base_addr,
    input wire [ADDR_WIDTH-1:0] mac_wscale_base_addr,
    input wire [REG_WIDTH-1:0] mac_scale_value,
    input wire [DATA_WIDTH-1:0] mac_data_zp,
    input wire [DATA_WIDTH-1:0] mac_weight_zp,
    input wire [DATA_WIDTH-1:0] mac_output_zp,
    input wire [IMAGE_WH-1:0] mac_num_line,
    input wire [IMAGE_WH-1:0] mac_start_y,
    input wire [IMAGE_COUNT-1:0] mac_num_outchannel,
    input wire [KE_NUM-1:0] mac_num_kernel_elmnt,
    input wire [BLOCK_WIDTH-1:0] mac_block_width,
    input wire [BLOCK_HEIGHT-1:0] mac_block_height,
    input wire [KERNEL_SIZE-1:0] mac_kernel_size,
    input wire [STRIDE-1:0] mac_stride,
// MAC-DepthConv
    input wire [IMAGE_COUNT-1:0] mac_num_image,
    input wire [NUM_GROUP-1:0] mac_num_group,
	input wire [1:0] mac_group_mode,
    input wire [OUTBLOCK_WIDTH-1:0] mac_outblock_width,
// MAC-MatMul
    input wire [BLOCK_HEIGHT-1:0] mac_num_r,
// QLinearAdd
    output wire qla_done,
    input wire qla_start,
    input wire [ADDR_WIDTH-1:0] qla_a_addr,
    input wire [ADDR_WIDTH-1:0] qla_b_addr,
    input wire [ADDR_WIDTH-1:0] qla_out_addr,
    input wire [NUM_DATA-1:0] qla_num_data,
    input wire [REG_WIDTH-1:0] qla_scale_ac,
    input wire [REG_WIDTH-1:0] qla_scale_bc,
    input wire [DATA_WIDTH-1:0] qla_a_zp,
    input wire [DATA_WIDTH-1:0] qla_b_zp,
    input wire [DATA_WIDTH-1:0] qla_out_zp,
*/

//AXI Lite Signals
    input  wire                                 s00_axi_aclk,
    input  wire                                 s00_axi_aresetn,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0]      s00_axi_awaddr,
    input  wire [2:0]                           s00_axi_awprot,
    input  wire                                 s00_axi_awvalid,
    output wire                                 s00_axi_awready,
    input  wire [C_S00_AXI_DATA_WIDTH-1:0]      s00_axi_wdata,
    input  wire [(C_S00_AXI_DATA_WIDTH/8)-1:0]  s00_axi_wstrb,
    input  wire                                 s00_axi_wvalid,
    output wire                                 s00_axi_wready,
    output wire [1:0]                           s00_axi_bresp,
    output wire                                 s00_axi_bvalid,
    input  wire                                 s00_axi_bready,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0]      s00_axi_araddr,
    input  wire [2:0]                           s00_axi_arprot,
    input  wire                                 s00_axi_arvalid,
    output wire                                 s00_axi_arready,
    output wire [C_S00_AXI_DATA_WIDTH-1:0]      s00_axi_rdata,
    output wire [1:0]                           s00_axi_rresp,
    output wire                                 s00_axi_rvalid,
    input  wire                                 s00_axi_rready,

// AXI Stream Signals
    // AXI Stream Slave
    input  wire                                  s_axis_aclk,
    input  wire                                  s_axis_aresetn,
    input  wire [C_S_AXIS_TDATA_WIDTH-1:0]       s_axis_tdata,
    input  wire [(C_S_AXIS_TDATA_WIDTH/8)-1:0]   s_axis_tkeep,
    input  wire                                  s_axis_tlast,
    output wire                                  s_axis_tready,
    input  wire                                  s_axis_tvalid,
    
    // AXI Stream Master
    input  wire                                 m_axis_aclk,
    input  wire                                 m_axis_aresetn,
    output wire [C_M_AXIS_TDATA_WIDTH-1:0]      m_axis_tdata,
    output wire [(C_M_AXIS_TDATA_WIDTH/8)-1:0]  m_axis_tkeep,
    output wire                                 m_axis_tlast,
    input  wire                                 m_axis_tready,
    output wire                                 m_axis_tvalid,
	
	
	output wire done_interrupt
);

// AXI Lite Signals -----------------------------------------------------------
wire [C_S00_AXI_DATA_WIDTH-1:0] PM0;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM1;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM2;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM3;
wire DMA_read_start, DMA_write_start;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM6;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM7;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM8;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM9;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM10;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM11;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM12;
wire [C_S00_AXI_DATA_WIDTH-1:0] PM13;
wire logic_start;
wire qla_mode;//QLA_MODE_1004 *add

// DMA
wire mem_read_start, mem_write_start;
wire [BUFFER_ID_WIDTH-1:0] dma_read_buffer_id, dma_write_buffer_id;
wire [ADDR_WIDTH-1:0] dma_read_base_addr, dma_write_base_addr;
wire [NUM_DATA-1:0] dma_read_num_data, dma_write_num_data;
wire mem_read_done, mem_write_done, mem_read_op, mem_write_op;
// MAC-Conv3d, Conv1x1
wire mac_op_done;
wire mac_start, conv3d, conv1x1, depthconv, matmul, mac_bias_onoff, mac_norm_onoff;
wire [OPCODE_WIDTH-1:0] process_ops;
wire [BUFFER_ID_WIDTH-1:0] mac_read_db_id;
wire [BUFFER_ID_WIDTH-1:0] mac_write_db_id;
wire [OUTCH_OFFSET-1:0] mac_store_outch_offset;
wire [ADDR_WIDTH-1:0] mac_wb_base_addr;
wire [ADDR_WIDTH-1:0] mac_db_base_addr;
wire [ADDR_WIDTH-1:0] mac_store_base_addr;
wire [ADDR_WIDTH-1:0] mac_bias_base_addr;
wire [ADDR_WIDTH-1:0] mac_wzero_base_addr;
wire [ADDR_WIDTH-1:0] mac_wscale_base_addr;
wire [REG_WIDTH-1:0] mac_scale_value;
wire [DATA_WIDTH-1:0] mac_data_zp;
wire [DATA_WIDTH-1:0] mac_weight_zp;
wire [DATA_WIDTH-1:0] mac_output_zp;
wire [IMAGE_WH-1:0] mac_num_line;
wire [IMAGE_WH-1:0] mac_start_y;
wire [IMAGE_COUNT-1:0] mac_num_outchannel;
wire [KE_NUM-1:0] mac_num_kernel_elmnt;
wire [BLOCK_WIDTH-1:0] mac_block_width;
wire [BLOCK_HEIGHT-1:0] mac_block_height;
wire [KERNEL_SIZE-1:0] mac_kernel_size;
wire [STRIDE-1:0] mac_stride;
// MAC-DepthConv
wire [IMAGE_COUNT-1:0] mac_num_image;
wire [NUM_GROUP-1:0] mac_num_group;
wire [1:0] mac_group_mode;
wire [OUTBLOCK_WIDTH-1:0] mac_outblock_width;
// MAC-MatMul
wire [BLOCK_HEIGHT-1:0] mac_num_r;
// QLinearAdd
wire qla_done;
wire qla_start, transpose_start;
wire [ADDR_WIDTH-1:0] qla_a_addr;
wire [ADDR_WIDTH-1:0] qla_b_addr;
wire [ADDR_WIDTH-1:0] qla_out_addr;
wire [NUM_DATA-1:0] qla_num_data;
wire [REG_WIDTH-1:0] qla_scale_ac;
wire [REG_WIDTH-1:0] qla_scale_bc;
wire [DATA_WIDTH-1:0] qla_a_zp;
wire [DATA_WIDTH-1:0] qla_b_zp;
wire [DATA_WIDTH-1:0] qla_out_zp;
wire [13:0] Num_Const; //QLA_MODE_1013 *Num_Const
//Transpose
wire trans_done;
//Zeropadding
wire zp_start, zp_done, zp_read_enable;
wire [IMAGE_WH-1:0] image_width, image_height;
wire [IMAGE_COUNT-1:0] num_channels;
wire [DATA_WIDTH-1:0] zero_point_value, w_start, w_end, h_start, h_end;


// MAC to BRAM Interface Signals --------------------------------------------------------------
wire dbuffer_write_en, dbuffer_read_en, bias_wread_en, mac_init_wread_en, mac_op_wread_en;
wire [ADDR_WIDTH-1:0] data_addr_rd, bias_update_w_raddr, mac_init_w_raddr, mac_op_w_raddr, save_addr_dbuffer;
wire [BUF_WIDTH-1:0] data_buf, mac_y_out;
    
// DMA Logic to BRAM Interface Signals -----------------------------------------------------------
wire [ADDR_WIDTH-1:0] dma_write_addr;
wire [ADDR_WIDTH-1:0] dma_read_addr;
wire [C_S_AXIS_TDATA_WIDTH-1:0] dma_write_data;
wire [C_M_AXIS_TDATA_WIDTH-1:0] dma_read_data;
wire dma_read_en, m_axis_tvalid_muxing, dma_write_en;

// BRAM Interface to BRAM Signals -----------------------------------------------------------------
wire [ADDR_WIDTH-1:0] addra_WB0, addra_DB0, addra_DB1, addrb_WB0, addrb_DB0, addrb_DB1;
wire [AXI_STREAM_DATA_WIDTH-1:0] dina_WB0, dina_DB0, dina_DB1;
wire wea_WB0, wea_DB0, wea_DB1, enb_WB0, enb_DB0, enb_DB1;
wire [AXI_STREAM_DATA_WIDTH-1:0] doutb_WB0, doutb_DB0, doutb_DB1;
wire wbuf_regceb, dbuf0_regceb, dbuf1_regceb;

//QLINEARADD Interface to BRAM Interface_0826_Add
wire QLinearAdd_enb, QLinearAdd_wea;
wire [ADDR_WIDTH-1:0] QLinearAdd_addrb;
wire [BUF_WIDTH-1:0] QLinearAdd_dina;
wire [ADDR_WIDTH-1:0] QLinearAdd_addra;

wire [BUF_WIDTH-1:0] QLinearAdd_out;//QLA_MODE_1004 *add
wire [BUF_WIDTH-1:0] QLinearAdd_Weight_out;//QLA_MODE_1004 *add
wire [ADDR_WIDTH-1:0] QLinearAdd_Weight_addrb;//QLA_MODE_1004 *add
wire QLinearAdd_Weight_enb;//QLA_MODE_1004 *add
wire [3:0] qla_opcode;//QLA_MODE_1004 *add
wire [1:0] qla_read_DB_id;//QLA_MODE_1004 *add
wire [1:0] qla_write_DB_id;//QLA_MODE_1004 *add
wire AddMul_Mode;

//Transpose Interface to BRAM Signals -----------------------------------------------------------------
wire trans_db_readen, trans_wb_writeen;
wire [ADDR_WIDTH-1:0] trans_db_addr_rd, trans_wb_addr_wr;
wire [BUF_WIDTH-1:0] trans_outdata;

// Zero padding Interface to BRAM Signals -----------------------------------------------------------------
wire zp_db_readen, zp_db_writeen;
wire [ADDR_WIDTH-1:0] zp_db_addr_rd, zp_wb_addr_wr;
wire [BUF_WIDTH-1:0] zp_outdata;


NPU_v1_0_S00_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
	) NPU_v1_0_S00_AXI_Inst (
	.S_AXI_ACLK(s00_axi_aclk),
	.S_AXI_ARESETN(s00_axi_aresetn),
	.S_AXI_AWADDR(s00_axi_awaddr),
	.S_AXI_AWPROT(s00_axi_awprot),
	.S_AXI_AWVALID(s00_axi_awvalid),
	.S_AXI_AWREADY(s00_axi_awready),
	.S_AXI_WDATA(s00_axi_wdata),
	.S_AXI_WSTRB(s00_axi_wstrb),
	.S_AXI_WVALID(s00_axi_wvalid),
	.S_AXI_WREADY(s00_axi_wready),
	.S_AXI_BRESP(s00_axi_bresp),
	.S_AXI_BVALID(s00_axi_bvalid),
	.S_AXI_BREADY(s00_axi_bready),
	.S_AXI_ARADDR(s00_axi_araddr),
	.S_AXI_ARPROT(s00_axi_arprot),
	.S_AXI_ARVALID(s00_axi_arvalid),
	.S_AXI_ARREADY(s00_axi_arready),
	.S_AXI_RDATA(s00_axi_rdata),
	.S_AXI_RRESP(s00_axi_rresp),
	.S_AXI_RVALID(s00_axi_rvalid),
	.S_AXI_RREADY(s00_axi_rready),

	.mem_read_done(mem_read_done), 
	.mem_write_done(mem_write_done), 
	.process_op_done(mac_op_done | qla_done | trans_done | zp_done),  //  mac_op_done | qla_done | transpose_done | .....; ************************************
	.PM0(PM0),
    .PM1(PM1),
    .PM2(PM2),
    .PM3(PM3),
    .DMA_read_start(DMA_read_start), 
	.DMA_write_start(DMA_write_start),
    .PM6(PM6),
    .PM7(PM7),
    .PM8(PM8),
    .PM9(PM9),
    .PM10(PM10),
    .PM11(PM11),
    .PM12(PM12),
    .PM13(PM13),
    .mem_read_op(mem_read_op), 
	.mem_write_op(mem_write_op), 
	.logic_start(logic_start)
	);

// AXI LITE WRAP
NPU_v1_0_S00_AXI_WRAP uut_AXI_Lite
(
    .clk(s00_axi_aclk),
    .resetn(s00_axi_aresetn),

    // Ports of Axi Slave Bus Interface S00_AXI
	.PM0(PM0),
    .PM1(PM1),
    .PM2(PM2),
    .PM3(PM3),
    .DMA_read_start(DMA_read_start), 
	.DMA_write_start(DMA_write_start),
    .PM6(PM6),
    .PM7(PM7),
    .PM8(PM8),
    .PM9(PM9),
    .PM10(PM10),
    .PM11(PM11),
    .PM12(PM12),
    .PM13(PM13),
	.logic_start(logic_start),

    // DMA
    .mem_read_start(mem_read_start), 
	.mem_write_start(mem_write_start),
	.dma_read_buffer_id(dma_read_buffer_id), 
	.dma_write_buffer_id(dma_write_buffer_id),
	.dma_read_base_addr(dma_read_base_addr), 
	.dma_write_base_addr(dma_write_base_addr),
	.dma_read_num_data(dma_read_num_data), 
	.dma_write_num_data(dma_write_num_data),

    // MAC-Conv3d, Conv1x1
    .mac_start(mac_start), 
    .process_ops(process_ops),
    .conv3d(conv3d), 
    .conv1x1(conv1x1), 
    .depthconv(depthconv), 
    .matmul(matmul), 
    .mac_bias_onoff(mac_bias_onoff), 
    .mac_norm_onoff(mac_norm_onoff),
    .mac_read_db_id(mac_read_db_id),
    .mac_write_db_id(mac_write_db_id),
    .mac_store_outch_offset(mac_store_outch_offset),
    .mac_wb_base_addr(mac_wb_base_addr),
    .mac_db_base_addr(mac_db_base_addr),
    .mac_store_base_addr(mac_store_base_addr),
    .mac_bias_base_addr(mac_bias_base_addr),
    .mac_wzero_base_addr(mac_wzero_base_addr),
    .mac_wscale_base_addr(mac_wscale_base_addr),
    .mac_scale_value(mac_scale_value),
    .mac_data_zp(mac_data_zp),
    .mac_weight_zp(mac_weight_zp),
    .mac_output_zp(mac_output_zp),
    .mac_num_line(mac_num_line),
    .mac_start_y(mac_start_y),
    .mac_num_outchannel(mac_num_outchannel),
    .mac_num_kernel_elmnt(mac_num_kernel_elmnt),
    .mac_block_width(mac_block_width),
    .mac_block_height(mac_block_height),
    .mac_kernel_size(mac_kernel_size),
    .mac_stride(mac_stride),
    // MAC-DepthConv
    .mac_num_image(mac_num_image),
    .mac_num_group(mac_num_group),
    .mac_group_mode(mac_group_mode),
    .mac_outblock_width(mac_outblock_width),
    // MAC-MatMul
    .mac_num_r(mac_num_r),
    // QLinearAdd
    .qla_start(qla_start),
    .transpose_start(transpose_start),
    .qla_a_addr(qla_a_addr),
    .qla_b_addr(qla_b_addr),
    .qla_out_addr(qla_out_addr),
    .qla_num_data(qla_num_data),
    .qla_scale_ac(qla_scale_ac),
    .qla_scale_bc(qla_scale_bc),
    .qla_a_zp(qla_a_zp),
    .qla_b_zp(qla_b_zp),
    .qla_out_zp(qla_out_zp),
	.qla_opcode(qla_opcode),//QLA_MODE_1004 *add
	.qla_read_DB_id(qla_read_DB_id),//QLA_MODE_1004 *add
	.qla_write_DB_id(qla_write_DB_id),//QLA_MODE_1004 *add
	.qla_mode(qla_mode),//QLA_MODE_1004 *add
	.Num_Const(Num_Const),
    .AddMul_Mode(AddMul_Mode),
    // Zeropadding
    .zp_start(zp_start),
    .image_width(image_width), 
    .image_height(image_height), 
    .num_channels(num_channels), 
    .zero_point_value(zero_point_value), 
    .w_start(w_start),  
    .w_end(w_end),  
    .h_start(h_start),  
    .h_end(h_end)
);
	
mac_top uut_mac_top
(
    .clk(s00_axi_aclk), 
    .resetn(s00_axi_aresetn),
    .start(mac_start),
    .conv3d(conv3d),
    .conv1x1(conv1x1),
    .depthconv(depthconv), 
    .matmul(matmul), 
    .Bias_onoff(mac_bias_onoff), 
    .Normal_onoff(mac_norm_onoff), 
    .Stride(mac_stride),
    .Kernel_size(mac_kernel_size),
    .group_mode(mac_group_mode),
    .Bias_addr(mac_bias_base_addr),
    .Weight_zero_addr(mac_wzero_base_addr),
    .Weight_scale_addr(mac_wscale_base_addr), 
    .Weight_addr_in(mac_wb_base_addr),
    .Data_addr_in(mac_db_base_addr),
    .Save_addr(mac_store_base_addr),
    .Save_addr_offset(mac_store_outch_offset),
    .Weight_zp(mac_weight_zp),
    .Block_width(mac_block_width),
    .Block_height(mac_block_height),
    .Out_block_width(mac_outblock_width), 
    .Image_y(mac_start_y),
    .Line_count(mac_num_line),
    .R_count(mac_num_r),
    .num_group(mac_num_group),
    .Image_count(mac_num_image),
    .Num_outchannels(mac_num_outchannel),
    .Num_kernel_elements(mac_num_kernel_elmnt),
    .Scale_value(mac_scale_value),
    .Data_zp(mac_data_zp),
    .Y_out_zp(mac_output_zp),
    
    .data_buf(data_buf),
    .weight_buf(doutb_WB0),
    .dbuffer_write_en(dbuffer_write_en),
    .dbuffer_read_en(dbuffer_read_en),
    .bias_wread_en(bias_wread_en), 
    .mac_init_wread_en(mac_init_wread_en), 
    .mac_op_wread_en(mac_op_wread_en),
    .data_addr_rd(data_addr_rd),
    .bias_update_w_raddr(bias_update_w_raddr), 
    .mac_init_w_raddr(mac_init_w_raddr), 
    .mac_op_w_raddr(mac_op_w_raddr), 
    .save_addr_dbuffer(save_addr_dbuffer),
    .y_out(mac_y_out),
    .done(mac_op_done)
);

// DMA Logic with AXI Stream Interface
NPU_v1_0_S_AXIS #
(
.ADDR_WIDTH(ADDR_WIDTH),
.C_S_AXIS_TDATA_WIDTH(C_S_AXIS_TDATA_WIDTH)
) NPU_v1_0_S_AXIS_Inst
(
	.start(mem_write_start),
	.dma_write_base_addr(dma_write_base_addr),
	.dma_write_done(mem_write_done),
		
	// BRAM Interface
    .dma_write_data(dma_write_data),
    .dma_write_en(dma_write_en),
	.dma_write_addr(dma_write_addr),

	// AXI4Stream sink: Clock
	.S_AXIS_ACLK(s_axis_aclk),
	.S_AXIS_ARESETN(s_axis_aresetn),
	.S_AXIS_TREADY(s_axis_tready),
	.S_AXIS_TDATA(s_axis_tdata),
	.S_AXIS_TKEEP(s_axis_tkeep),
	.S_AXIS_TLAST(s_axis_tlast),
	.S_AXIS_TVALID(s_axis_tvalid)
);

NPU_v1_0_M_AXIS #
(
.ADDR_WIDTH(ADDR_WIDTH),
.NUM_DATA(NUM_DATA),
.C_M_AXIS_TDATA_WIDTH(C_M_AXIS_TDATA_WIDTH)
) NPU_v1_0_M_AXIS_Inst
(
	.start(mem_read_start),
	.num_data(dma_read_num_data),
	.dma_read_base_addr(dma_read_base_addr),
	.dma_read_done(mem_read_done),

	// BRAM Interface
    .dma_read_data(dma_read_data),
    .dma_read_en(dma_read_en),
	.dma_read_addr(dma_read_addr),

	// Global ports
	.M_AXIS_ACLK(m_axis_aclk),
	.M_AXIS_ARESETN(m_axis_aresetn),
	.M_AXIS_TVALID(m_axis_tvalid),
	.M_AXIS_TDATA(m_axis_tdata),
	.M_AXIS_TKEEP(m_axis_tkeep),
	.M_AXIS_TLAST(m_axis_tlast),
	.M_AXIS_TREADY(m_axis_tready)
); 
    
QLinearAddMult
#(.A_BIT(A_BIT), .B_BIT(B_BIT), .C_BIT(C_BIT), .C_AXI_LITE_WIDTH(C_AXI_LITE_WIDTH), .ADDR_DEPTH(ADDR_WIDTH), .Scale(Scale), .ZP(ZP))
QLinearAddMul
(
    .clk(s00_axi_aclk),//o
    .resetn(s00_axi_aresetn),//o
    
    .Op_Code(process_ops),//o         
    .Read_DB_ID(mac_read_db_id),//o      
    .Write_DB_ID(mac_write_db_id),//o     
    .Base_Addr(qla_a_addr),//o       
    .Base_Addr2(qla_b_addr),//o                    
    .Save_Addr(qla_out_addr),  
    .Num_Data(qla_num_data),//o        
    .A_ZP(qla_a_zp),//o      
    .B_ZP(qla_b_zp),//o     
    .C_ZP(qla_out_zp),//o
    .A_Scale(qla_scale_ac),//o         
    .B_Scale(qla_scale_bc),//o   
	.qla_mode(qla_mode),//QLA_MODE_1004 *add	
	.Num_Const(Num_Const),
	.AddMul_Mode(AddMul_Mode),
    
    .Data(data_buf),//o
    .Data_addrb(QLinearAdd_addrb),//o
    .Data_enb(QLinearAdd_enb),//o
    .C(QLinearAdd_dina),//o
    .C_valid(QLinearAdd_wea),//o
    .C_addra(QLinearAdd_addra),//o
    
    .run_pulse(qla_start),//o
    .done_pulse(qla_done),//o

	.Weight(QLinearAdd_Weight_out),//o //QLA_MODE_1004 *add
	.Weight_addrb(QLinearAdd_Weight_addrb),//o //QLA_MODE_1004 *add
	.Weight_enb(QLinearAdd_Weight_enb)//o //QLA_MODE_1004 *add
	
);
    
transpose transpose_Inst
(
    .clk(s00_axi_aclk),
    .resetn(s00_axi_aresetn),
    .start(transpose_start),
    .data_addr_in(qla_a_addr), 
    .data_addr_offset(qla_b_addr), 
    .save_addr_in(qla_out_addr), 
    .num_data(qla_num_data),
    .data_buf(data_buf),
    .dbuffer_read_en(trans_db_readen),
    .wbuffer_write_en(trans_wb_writeen), 
    .done(trans_done),
    .data_addr_rd(trans_db_addr_rd), 
    .weight_addr_write(trans_wb_addr_wr),
    .traspose_output(trans_outdata)
);

zeropadding zeropadding_Inst
(
    .clk(s00_axi_aclk),
    .resetn(s00_axi_aresetn),
    .start(zp_start),
    .image_width(image_width), 
    .image_height(image_height), 
    .num_channels(num_channels), 
    .zp_val(zero_point_value), 
    .w_start(w_start),  
    .w_end(w_end),  
    .h_start(h_start),  
    .h_end(h_end), 
    .data_addr_in(qla_a_addr),  
    .save_addr_in(qla_b_addr), 
    .data_buf(data_buf), 
    .dbuffer_read_en(zp_db_readen),  
    .data_write_en(zp_db_writeen), 
    .zp_read_enable(zp_read_enable),
    .done(zp_done), 
    .data_addr_rd(zp_db_addr_rd),  
    .data_addr_write(zp_wb_addr_wr), 
    .zp_output(zp_outdata)
);

bram_interface uut_bram_interface
(
    .clk(s00_axi_aclk), 
    .resetn(s00_axi_aresetn), 
    .mem_read_op(mem_read_op),
    .mem_write_op(mem_write_op),
    .dma_read_enable(m_axis_tready),
    .zp_read_enable(zp_read_enable),
    .process_ops(process_ops),
    .bias_wread_en(bias_wread_en),
    .mac_init_wread_en(mac_init_wread_en), 
    .mac_op_wread_en(mac_op_wread_en),
    .d_mac_op_re(dbuffer_read_en), 
    .dma_re(dma_read_en),
    .qlinearadd_re(QLinearAdd_enb), 
    .transpose_re(trans_db_readen), 
    .zp_re(zp_db_readen),
    //.qlinearmul_re(1'b0), 
    .dma_we(dma_write_en), 
    .mac_op_we(dbuffer_write_en), 
    .qlinearadd_we(QLinearAdd_wea), 
    .transpose_we(trans_wb_writeen), 
    .zp_we(zp_db_writeen),
    //.qlinearmul_we(1'b0), 
    .dma_read_buffer_id(dma_read_buffer_id),
    .dma_write_buffer_id(dma_write_buffer_id),
    .pro_read_db_id(mac_read_db_id), 
    .pro_write_db_id(mac_write_db_id), 
    .bias_update_w_raddr(bias_update_w_raddr), 
    .mac_init_w_raddr(mac_init_w_raddr), 
    .mac_op_w_raddr(mac_op_w_raddr),
    .d_mac_op_raddr(data_addr_rd), 
    .dma_raddr(dma_read_addr), 
    .qlinearadd_raddr(QLinearAdd_addrb), 
    .transpose_raddr(trans_db_addr_rd), 
    .zp_raddr(zp_db_addr_rd),
    //.qlinearmul_raddr(16'h0), 
    .dma_waddr(dma_write_addr), 
    .mac_op_waddr(save_addr_dbuffer), 
    .qlinearadd_waddr(QLinearAdd_addra), 
    .transpose_waddr(trans_wb_addr_wr), 
    .zp_waddr(zp_wb_addr_wr),
    //.qlinearmul_waddr(16'h0), 
    .dma_data(dma_write_data), 
    .mac_op_data(mac_y_out), 
    .qlinearadd_data(QLinearAdd_dina), 
    .transpose_data(trans_outdata), 
    .zp_data(zp_outdata),
    //.qlinearmul_data(256'h0), 
    .processor_db_out(data_buf), 
    .mem_db_out(dma_read_data),

    .doutb_WB0(doutb_WB0), 
    .doutb_DB0(doutb_DB0), 
    .doutb_DB1(doutb_DB1),
    .w_waddr(addra_WB0), 
    .w_raddr(addrb_WB0), 
    .d0_waddr(addra_DB0), 
    .d1_waddr(addra_DB1), 
    .d0_raddr(addrb_DB0), 
    .d1_raddr(addrb_DB1),
    .w_data(dina_WB0), 
    .d0_data(dina_DB0), 
    .d1_data(dina_DB1), 
    .w_we(wea_WB0),
	
    .d0_we(wea_DB0), 
    .d1_we(wea_DB1), 
	
    .wbuf_read_enable(enb_WB0), 
    .dbuf0_read_enable(enb_DB0), 
    .dbuf1_read_enable(enb_DB1),
	
	.qla_weight_out(QLinearAdd_Weight_out),//QLA_MODE_1004 *add
	.qla_weight_raddr(QLinearAdd_Weight_addrb),//QLA_MODE_1004 *add
	.qla_weight_enb(QLinearAdd_Weight_enb),//QLA_MODE_1004 *add
	.qla_mode(qla_mode)//QLA_MODE_1004 *add
);

BRAM_RW BRAM
(
    .clk(s00_axi_aclk),
    .addra_WB0(addra_WB0),
    .addra_DB0(addra_DB0),
    .addra_DB1(addra_DB1),
    .addrb_WB0(addrb_WB0),
    .addrb_DB0(addrb_DB0),
    .addrb_DB1(addrb_DB1),
    .dina_WB0(dina_WB0),
    .dina_DB0(dina_DB0),
    .dina_DB1(dina_DB1),
    .wea_WB0(wea_WB0),
    .wea_DB0(wea_DB0),
    .wea_DB1(wea_DB1),
    .enb_WB0(enb_WB0),
    .enb_DB0(enb_DB0),
    .enb_DB1(enb_DB1),
    .doutb_WB0(doutb_WB0),
    .doutb_DB0(doutb_DB0),
    .doutb_DB1(doutb_DB1)
);
assign done_interrupt=mac_op_done | qla_done | trans_done | zp_done;
    
endmodule 

