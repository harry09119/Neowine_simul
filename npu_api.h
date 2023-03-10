#ifndef SRC_NPU_API
#define SRC_NPU_API

#define DMA_DEV_ID		    XPAR_AXIDMA_0_DEVICE_ID

#define CONV3D 8
#define CONV1X1 9
#define DEPTHCONV 10
#define MATMUL 11
#define QLINEARADD 12
#define TRANSPOSE 15
#define WEIGHT_BUFFER 1
#define DATA_BUFFER_0 2
#define DATA_BUFFER_1 3
#define ARRAYSIZE 32

void initialize_DMA();
void npu_mem_write(char buf_id, short base_addr, u32 num_data, u32 *data);
void npu_mem_read(char buf_id, short base_addr, u32 num_data, u32 *data);
void npu_conv_test(char op_code, char read_buf_id, char write_buf_id, short num_outchannel, short store_channel_addr_offset,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, u32 scale_value, char data_zeropoint,
		char weight_zeropoint, char output_zeropoint, char bias_onoff, char normal_onoff, short block_height,
		char block_width, short num_line, short start_y, int num_kernel_element, char kernel_size, char stride);

void npu_conv(char op_code, char read_buf_id, char write_buf_id, short num_inchannel, short num_outchannel,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr, short weight_zeropoint_base_addr,
		short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint, char weight_zeropoint, char output_zeropoint,
		char bias_onoff, char normal_onoff, short image_height, char image_width, short start_y, short num_line, char kernel_size, char stride);

void npu_depthconv_test(char op_code, char read_buf_id, char write_buf_id, short out_block_width, short store_channel_addr_offset,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint,
		char weight_zeropoint, char output_zeropoint, char bias_onoff, char normal_onoff, short block_height,
		char block_width, short num_line, short num_image, short num_group, char group_mode, char kernel_size, char stride);

void npu_depthconv(char op_code, char read_buf_id, char write_buf_id,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint, char weight_zeropoint, char output_zeropoint,
		char bias_onoff, char normal_onoff, short image_height, char image_width, short num_image, short num_group, char group_mode, char kernel_size, char stride);

void npu_matmul(char op_code, char read_buf_id, char write_buf_id,
		short weight_buf_base_addr, short read_buf_read_base_addr,
		short read_buf_store_base_addr, short num_M, short num_R, short num_N,
		u32 scale_value, char b_zeropoint, char a_zeropoint, char output_zeropoint);

void npu_qla(char op_code, char read_buf_id, char write_buf_id,
	char output_ZP, short A_Base_Addr, short B_Base_Addr, short output_Base_Addr,
	short Num_Data, char A_ZP, char B_ZP, int AC_Scale, int BC_Scale);

void npu_transpose(char op_code, char read_buf_id, char write_buf_id, short input_base_addr, short Output_base_addr, short input_offset_addr, int num_indata);

#endif /* SRC_NPU_API */
