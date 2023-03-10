#include "xil_printf.h"
#include "xaxidma.h"
#include "npu_api.h"
#include "NPU.h"
#include "xtime_l.h"

/************************** Variable Definitions *****************************/
XAxiDma AxiDma; //DMA device instance definition

void initialize_DMA()
{
	XAxiDma_Config *CfgPtr;
	int Status;

	/* Initialize the XAxiDma device.
	 */
	CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
	if (!CfgPtr) {
		xil_printf("No config found for %d\r\n", DMA_DEV_ID);
		return XST_FAILURE;
	}

	Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
	if (Status != XST_SUCCESS) {
		xil_printf("Initialization failed %d\r\n", Status);
		return XST_FAILURE;
	}

	if(XAxiDma_HasSg(&AxiDma)){
		xil_printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Disable interrupts, we use polling mode
	 */
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
						XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
						XAXIDMA_DMA_TO_DEVICE);
	return;
}

void npu_mem_write(char buf_id, short base_addr, u32 num_data, u32 *data)
{
	int Status;
	u32 reg;

	Xil_DCacheFlushRange((INTPTR)data, num_data*sizeof(int));

	// Data transfer
	Status = XAxiDma_SimpleTransfer(&AxiDma,(u32) data, num_data*sizeof(int), XAXIDMA_DMA_TO_DEVICE);

	if (Status != XST_SUCCESS) {
		xil_printf("DMA write Operations Failed\r\n");
		return XST_FAILURE;
	}

	/* NPU Register Setup */
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG2_OFFSET, (u32)(buf_id));
	reg = ((u32)base_addr << 16) + num_data/8;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG3_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG5_OFFSET, 1);

	while (XAxiDma_Busy(&AxiDma,XAXIDMA_DMA_TO_DEVICE)) {
				/* Wait */
	}

	//while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x10) != 0x10) {
	//		/* Wait */
	//}

	/* Test finishes successfully*/
	return;
}

void npu_mem_read(char buf_id, short base_addr, u32 num_data, u32 *data)
{
	int Status;
	u32 reg;

	Xil_DCacheFlushRange((INTPTR)data, num_data*sizeof(int));

	// Data transfer
	Status = XAxiDma_SimpleTransfer(&AxiDma,(u32) data, num_data*sizeof(int), XAXIDMA_DEVICE_TO_DMA);

	if (Status != XST_SUCCESS) {
		xil_printf("DMA read Operations Failed\r\n");
		return XST_FAILURE;
	}

	/* NPU Register Setup */
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG0_OFFSET, (u32)(buf_id));
	reg = ((u32)base_addr << 16) + num_data/8;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG1_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG4_OFFSET, 1);

	while (XAxiDma_Busy(&AxiDma,XAXIDMA_DEVICE_TO_DMA)) {
				/* Wait */
	}

	//while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x1) != 0x1) {
	//		/* Wait */
	//}

	// invalidate the cache to avoid reading garbage
	Xil_DCacheInvalidateRange((INTPTR)data, num_data*sizeof(int));

	/* Test finishes successfully */
	return;
}

void npu_conv_test(char op_code, char read_buf_id, char write_buf_id, short num_outchannel, short store_channel_addr_offset,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, u32 scale_value, char data_zeropoint,
		char weight_zeropoint, char output_zeropoint, char bias_onoff, char normal_onoff, short block_height,
		char block_width, short num_line, short start_y, int num_kernel_element, char kernel_size, char stride)
{
	// op_code : conv3d = 8, conv1x1 = 9
	XTime tStart, tEnd;
	long long HW_time;
	int Status;
	u32 reg;

	//XTime_GetTime(&tStart);

	// Register Set----------
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24) + ((u32)num_outchannel << 10) + (u32)store_channel_addr_offset;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)weight_buf_base_addr << 16) + (u32)read_buf_read_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)read_buf_store_base_addr << 16) + (u32)bias_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)weight_zeropoint_base_addr << 16) + (u32)weight_scale_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, scale_value);
	reg = ((u32)data_zeropoint << 24) + ((u32)weight_zeropoint << 16) + ((u32)output_zeropoint << 8) + ((u32)bias_onoff << 1) + (u32)normal_onoff;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);
	reg = ((u32)block_height << 16) + ((u32)block_width << 10) + (u32)num_line;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, reg);
	reg = ((u32)start_y << 22) + ((u32)num_kernel_element << 4) + ((u32)kernel_size << 2) + (u32)stride;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, reg);

	//XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}

	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}

void npu_conv(char op_code, char read_buf_id, char write_buf_id, short num_inchannel, short num_outchannel,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr, short weight_zeropoint_base_addr,
		short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint, char weight_zeropoint, char output_zeropoint,
		char bias_onoff, char normal_onoff, short image_height, char image_width, short start_y, short num_line, char kernel_size, char stride)
{
	// op_code : conv3d = 8, conv1x1 = 9
	XTime tStart, tEnd;
	long long HW_time;
	int Status;
	u32 reg;

	if (op_code == 9) kernel_size = 1;

	char block_width;
	short mod, output_width, output_height, store_channel_addr_offset, block_height;
	int num_kernel_element;

	output_width = (image_width - kernel_size) / stride + 1;
	output_height = (image_height - kernel_size) / stride + 1;

	if ((output_width % 32) != 0) mod = 1;
	else mod = 0;

	store_channel_addr_offset = (output_width / 32 + mod) * output_height;
	num_kernel_element = kernel_size * kernel_size * num_inchannel;

	if ((image_width % 32) != 0) mod = 1;
	else mod = 0;

	block_width = (char)(image_width / 32 + mod);
	block_height = image_height * (short)block_width;

	if (kernel_size == 3) kernel_size = 1;
	else if (kernel_size == 5) kernel_size = 2;
	else if (kernel_size == 7) kernel_size = 3;

	if (op_code == 9) kernel_size = 0;

	//XTime_GetTime(&tStart);

	// Register Set----------
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24) + ((u32)num_outchannel << 10) + (u32)store_channel_addr_offset;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)weight_buf_base_addr << 16) + (u32)read_buf_read_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)read_buf_store_base_addr << 16) + (u32)bias_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)weight_zeropoint_base_addr << 16) + (u32)weight_scale_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, scale_value);
	reg = ((u32)data_zeropoint << 24) + ((u32)weight_zeropoint << 16) + ((u32)output_zeropoint << 8) + ((u32)bias_onoff << 1) + (u32)normal_onoff;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);
	reg = ((u32)block_height << 16) + ((u32)block_width << 10) + (u32)num_line;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, reg);
	reg = ((u32)start_y << 22) + ((u32)num_kernel_element << 4) + ((u32)kernel_size << 2) + (u32)stride;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, reg);

	//XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}

	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}

void npu_depthconv_test(char op_code, char read_buf_id, char write_buf_id, short out_block_width, short store_channel_addr_offset,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint,
		char weight_zeropoint, char output_zeropoint, char bias_onoff, char normal_onoff, short block_height,
		char block_width, short num_line, short num_image, short num_group, char group_mode, char kernel_size, char stride)
{
	XTime tStart, tEnd;
	int HW_time, Status;
	u32 reg;

	//XTime_GetTime(&tStart);
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24) + ((u32)out_block_width << 10) + (u32)store_channel_addr_offset;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)weight_buf_base_addr << 16) + (u32)read_buf_read_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)read_buf_store_base_addr << 16) + (u32)bias_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)weight_zeropoint_base_addr << 16) + (u32)weight_scale_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, scale_value);
	reg = ((u32)data_zeropoint << 24) + ((u32)weight_zeropoint << 16) + ((u32)output_zeropoint << 8) + ((u32)bias_onoff << 1) + (u32)normal_onoff;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);
	reg = ((u32)block_height << 16) + ((u32)block_width << 10) + (u32)num_line;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, reg);
	reg = ((u32)num_image << 20) + ((u32)num_group << 6) + ((u32)group_mode << 4) + ((u32)kernel_size << 2) + (u32)stride;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, reg);

	//XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}
	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}

void npu_depthconv(char op_code, char read_buf_id, char write_buf_id,
		short weight_buf_base_addr, short read_buf_read_base_addr, short read_buf_store_base_addr, short bias_base_addr,
		short weight_zeropoint_base_addr, short weight_scale_base_addr, unsigned int scale_value, char data_zeropoint, char weight_zeropoint, char output_zeropoint,
		char bias_onoff, char normal_onoff, short image_height, char image_width, short num_image, short num_group, char group_mode, char kernel_size, char stride)
{
	XTime tStart, tEnd;
	int HW_time, Status;
	u32 reg;
	char block_width;
	short mod, num_line, output_width, output_height, out_block_width, store_channel_addr_offset, block_height;

	num_line = image_height;

	output_width = (image_width - kernel_size) / stride + 1;
	output_height = (image_height - kernel_size) / stride + 1;

	if ((output_width % 32) != 0) mod = 1;
	else mod = 0;

	out_block_width = output_width / 32 + mod;
	store_channel_addr_offset = out_block_width * output_height;

	if ((image_width % 32) != 0) mod = 1;
	else mod = 0;

	block_width = (char)(image_width / 32 + mod);
	block_height = image_height * (short)block_width;

	if (kernel_size == 3) kernel_size = 1;
	else if (kernel_size == 5) kernel_size = 2;
	else if (kernel_size == 7) kernel_size = 3;

	//XTime_GetTime(&tStart);
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24) + ((u32)out_block_width << 10) + (u32)store_channel_addr_offset;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)weight_buf_base_addr << 16) + (u32)read_buf_read_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)read_buf_store_base_addr << 16) + (u32)bias_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)weight_zeropoint_base_addr << 16) + (u32)weight_scale_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, scale_value);
	reg = ((u32)data_zeropoint << 24) + ((u32)weight_zeropoint << 16) + ((u32)output_zeropoint << 8) + ((u32)bias_onoff << 1) + (u32)normal_onoff;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);
	reg = ((u32)block_height << 16) + ((u32)block_width << 10) + (u32)num_line;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, reg);
	reg = ((u32)num_image << 20) + ((u32)num_group << 6) + ((u32)group_mode << 4) + ((u32)kernel_size << 2) + (u32)stride;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, reg);

	//XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}
	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}

void npu_matmul(char op_code, char read_buf_id, char write_buf_id,
		short weight_buf_base_addr, short read_buf_read_base_addr,
		short read_buf_store_base_addr, short num_M, short num_R, short num_N,
		u32 scale_value, char b_zeropoint, char a_zeropoint, char output_zeropoint)
{
	XTime tStart, tEnd;
	int HW_time, Status;
	u32 reg;
	short mod, numofN;

	if ((num_N % 32) != 0) mod = 1;
	else mod = 0;
	numofN = num_N / 32 + mod;

	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)weight_buf_base_addr << 16) + (u32)read_buf_read_base_addr;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)read_buf_store_base_addr << 16) + (u32)num_M;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)num_R << 16) + (u32)numofN;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, scale_value);
	reg = ((u32)b_zeropoint << 24) + ((u32)a_zeropoint << 16) + ((u32)output_zeropoint << 8);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, 0);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, 0);


	XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}
	XTime_GetTime(&tEnd);
	HW_time = 2*(tEnd - tStart);
	//printf(" Count per second %d\n", COUNTS_PER_SECOND);
	printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}

void npu_qla(char op_code, char read_buf_id, char write_buf_id,
	char output_ZP, short A_Base_Addr, short B_Base_Addr, short output_Base_Addr,
	short Num_Data, char A_ZP, char B_ZP, int AC_Scale, int BC_Scale)
{
	u32 reg;
	int Status;
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24) + ((u32)output_ZP);
	//xil_printf("%x\n",reg);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)A_Base_Addr << 16) + ((u32)B_Base_Addr);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)output_Base_Addr << 16) + ((u32)Num_Data);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);
	reg = ((u32)A_ZP << 16) + ((u32)B_ZP);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, reg);
	reg = ((u32)AC_Scale);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, reg);
	reg = ((u32)BC_Scale);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, reg);



	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);//start

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
	}

	return;
}

void npu_transpose(char op_code, char read_buf_id, char write_buf_id, short input_base_addr, short Output_base_addr, short input_offset_addr, int num_indata)
{
	// op_code : conv3d = 8, conv1x1 = 9
	XTime tStart, tEnd;
	long long HW_time;
	int Status;
	u32 reg;
	short offset_mod, mod;

	if ((input_offset_addr % 32) != 0) mod = 1;
	else mod = 0;
	offset_mod = input_offset_addr / 32 + mod;

	//XTime_GetTime(&tStart);

	// Register Set----------
	reg = ((u32)op_code << 28) + ((u32)read_buf_id << 26) + ((u32)write_buf_id << 24);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG6_OFFSET, reg);
	reg = ((u32)input_base_addr << 16) + (u32)offset_mod;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG7_OFFSET, reg);
	reg = ((u32)Output_base_addr << 16) + (u32)num_indata;
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG8_OFFSET, reg);

	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG9_OFFSET, 0);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG10_OFFSET, 0);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG11_OFFSET, 0);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG12_OFFSET, 0);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG13_OFFSET, 0);

	//XTime_GetTime(&tStart);
	NPU_mWriteReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG14_OFFSET, 1);

	while (((Status = NPU_mReadReg(XPAR_NPU_0_S00_AXI_BASEADDR, NPU_S00_AXI_SLV_REG15_OFFSET)) & 0x100) == 0x100) {
			/* Wait */
	}

	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

	/* Test finishes successfully*/
	return;
}
