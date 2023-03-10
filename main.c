#include <stdio.h>
#include "xaxidma.h"
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "npu_api.h"
#include "npu_sw_func.h"
#include "NPU.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xtime_l.h"

int main(){

	XTime tStart, tEnd;
	long long SW_time, HW_time;
	//unsigned int input_data[2752], input_weight[2112], output_data[4672]; // Each Buffer Size : 1MB = 131,072x(u32)
	unsigned int input_data[200000], input_weight[2112], output_data[200000];
	int i, j;
	int Status;

	xil_printf("start\n");
    init_platform();

    //FILE *fptest;
    //fptest = fopen("InitDataBuffer-vitis", "rb");
    //fread(input_data, sizeof(int), 1032, fptest); 또는 fscanf();
    // File System 부재로 fscanf가 동작하지 않음...
    //1. Debugging 모드로 들어가서 input_data의 address에 "InitDataBuffer-vitis.bin" binary data file을 import하여 로딩한다. ######
    //2. input_weight의 address에 "InitWeightBuffer-merge-vitis.bin" binary data file을 import하여 로딩한다. ######
    //   Conv3d, Conv1x1, DepthConv, Matmul용 weight값이 통합되어 있다. 크기는 2배 이상임
    //   Conv3d, Conv1x1용 Weight data base address는 0, DepthConv, Matmul용 data base address는 108.
    //3. 아래 operation 중 실행하고자하는 것을 선택하여 주석을 푼다.
    // ********* 최종적으로 아래와 같이 프로그램으로 데이터를 생성하여 사용함.

	// initialize DMA
	initialize_DMA();

	// Data Setup ----------------------------------------------------
/*	int src_width = 43;
	int src_height = 43;
	int src[7396], dst[11008];

	// 43x43x4 = 7396 ------------------------------
	for (i = 0; i < 7396; i++) {
		src[i] = i % 128;
	}
	// 64x43x4 = 11008
	for (i = 0; i < 11008; i++) {
		dst[i] = 0;
	}

	// Source data Line packing
	int* dst_index;
	int* src_index;
	dst_index = dst;
	src_index = src;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < src_height; j++) {
			memcpy(dst_index, src_index, sizeof(int) * src_width);
			src_index += src_width;
			dst_index += 64;
		}
	}

	unsigned char chartemp[11008];
	for (i = 0; i < 11008; i++) {
		chartemp[i] = (char)dst[i];
	}

	unsigned int* testint;
	testint = (unsigned int*)chartemp;

	for (i = 0; i < 2752; i++) {
		input_data[i] = testint[i];
	}
	// Data Setup ----------------------------------------------------

	// Weight Setup for Depth----------------------------------------------------
	//Conv3d, Conv1x1------
	for (i = 0; i < 2592; i++) { //81 x 32 = 2592
		chartemp[i] = (char)src[i];
	}

	testint = (unsigned int*)chartemp;
	for (i = 0; i < 648; i++) {
		input_weight[i] = testint[i];
	}

	char zp_values[96];
	for (i = 0; i < 96; i++) {
		zp_values[i] = 16; // 임의의 값
	}

	testint = (unsigned int*)zp_values;
	for (i = 0; i < 24; i++) {
		input_weight[648+i] = testint[i];
	}

	unsigned int bias_values[96];
	bias_values[0] = 10000000;

	for (i = 1; i < 96; i++) {
		bias_values[i] = bias_values[i - 1] + 1000; // 임의의 값
	}

	for (i = 0; i < 96; i++) {
		input_weight[672 + i] = bias_values[i];
	}

	unsigned int scale_values[96];
	scale_values[0] = 53021371;
	for (int n = 1; n < 96; n++) {
		scale_values[n] = scale_values[n - 1] + 1000;// 임의의 값
	}

	for (i = 0; i < 96; i++) {
		input_weight[768 + i] = scale_values[i];
	}

	// DepthConv, Matmul-----
	for (i = 0; i < 4704; i++) { //81 x 32 = 2592
		chartemp[i] = (char)src[i];
	}

	testint = (unsigned int*)chartemp;
	for (i = 0; i < 1176; i++) {
		input_weight[864+i] = testint[i];
	}

	testint = (unsigned int*)zp_values;
	for (i = 0; i < 8; i++) {
		input_weight[2040+i] = testint[i];
	}

	for (i = 0; i < 32; i++) {
		input_weight[2048 + i] = bias_values[i];
	}

	for (i = 0; i < 32; i++) {
		input_weight[2080 + i] = scale_values[i];
	}

	//for (i = 0; i < 2112; i=i+8) {
	//	xil_printf("%08x%08x%08x%08x%08x%08x%08x%08x\n", input_weight[i+7],input_weight[i+6],input_weight[i+5],input_weight[i+4],
	//			input_weight[i+3],input_weight[i+2],input_weight[i+1],input_weight[i]);
	//	}

	j = 0;
	char weight[81][ARRAYSIZE];
	for(i=0; i < 81; i++) { // 27 = 3x3x3, 27x3(32ea) = 81
		for(int n=0;n<ARRAYSIZE;n++) {
			weight[i][n] = j%128;
			j++;
		}
	}
*/
	char *dest;
	dest = (char *)input_data;

	// Weight Setup for Depth----------------------------------------------------




	//1) DMA Test ------------------------------------------------
/*    npu_mem_write(DATA_BUFFER_0, 0, 2064, input_data); // Export data Length : 8256
    npu_mem_read(DATA_BUFFER_0, 0, 2064, output_data);

    for (i = 0; i < 2064; i++) {
    	    if(input_data[i] != output_data[i])
    	    	printf("input: %d, output: %d, Different Results!\n", input_data[i], output_data[i]);
    	    //else printf("input: %d, output: %d\n", input_data[i], output_data[i]);
    	 }

	for (i = 0; i < 200000; i++) {
		input_data[i] = i;
	}

	npu_mem_write(DATA_BUFFER_0, 0, 65528, input_data); // Export data Length : 8256
	npu_mem_read(DATA_BUFFER_0, 0, 65528, output_data);

	for (i = 0; i < 65528; i++) {
	    if(input_data[i] != output_data[i])
	    	printf("index: %d, input: %d, output: %d, Different Results!\n", i, input_data[i], output_data[i]);
	    //else printf("input: %d, output: %d\n", input_data[i], output_data[i]);
	 }
	 */
    //2) Conv3d Test ---------------------------------------------
/*
	XTime_GetTime(&tStart);
	npu_conv3d_sw(dest, weight, zp_values, bias_values, scale_values, 32, 128, 1, 1);
	XTime_GetTime(&tEnd);


	SW_time = 2*(tEnd - tStart);
	printf("SW Output took %lld clock cycles. %lf mseconds\r\n", SW_time, (double)SW_time/(double)1200000);

	XTime_GetTime(&tStart);
    npu_mem_write(DATA_BUFFER_0, 0, 2064, input_data); // import "InitDataBuffer-vitis.bin", Import data Length : 8256
    //npu_mem_write(WEIGHT_BUFFER, 0, 864, input_weight); // import "InitWeightBuffer-vitis.bin", Import data Length : 3456, // for Time check, full image
    npu_mem_write(WEIGHT_BUFFER, 0, 2112, input_weight); // import "InitWeightBuffer-merge-vitis.bin", Import data Length : 8448
    //npu_conv(CONV3D, DATA_BUFFER_0, DATA_BUFFER_1, 73, 8, 0, 0, 0, 84, 81, 96, 0, 32, 16, 128, 1, 1, 86, 2, 41, 0, 27, 1, 1); // for Time check, full image

    //XTime_GetTime(&tStart);
    //npu_conv_test(CONV3D, DATA_BUFFER_0, DATA_BUFFER_1, 73, 8, 0, 0, 0, 84, 81, 96, 0, 32, 16, 128, 1, 1, 86, 2, 4, 0, 27, 1, 1);
    npu_conv(CONV3D, DATA_BUFFER_0, DATA_BUFFER_1, 73, 73, 0, 0, 0, 84, 81, 96, 0, 32, 16, 128, 1, 1, 43, 43, 0, 4, 1, 1);
	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

    npu_mem_read(DATA_BUFFER_1, 0, 4672, output_data);  // Export data Length : 18688

	XTime_GetTime(&tEnd);
	HW_time = 2*(tEnd - tStart);
	//printf(" Count per second %d\n", COUNTS_PER_SECOND);
	printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);
*/
    //3) Conv1x1 Test ---------------------------------------------
	/*XTime_GetTime(&tStart);

	XTime_GetTime(&tEnd);
	SW_time = tEnd - tStart;
	xil_printf("SW Output took %d clock cycles.\n", SW_time);*/
/*
    npu_mem_write(DATA_BUFFER_0, 0, 2064, input_data);
    npu_mem_write(WEIGHT_BUFFER, 0, 2112, input_weight);
    //npu_conv_test(CONV1X1, DATA_BUFFER_0, DATA_BUFFER_1, 73, 4, 0, 0, 0, 84, 81, 96, 0, 32, 16, 128, 1, 1, 86, 2, 2, 0, 3, 1, 1);
    npu_conv(CONV1X1, DATA_BUFFER_0, DATA_BUFFER_1, 73, 73, 0, 0, 0, 84, 81, 96, 0, 32, 16, 128, 1, 1, 43, 43, 0, 2, 1, 1);
    npu_mem_read(DATA_BUFFER_1, 0, 2336, output_data);  // Export data Length : 9344
*/

    //4) DepthConv Test ---------------------------------------------
    //char W_zp = 16;
    //unsigned int Scale_value = 53021371;
/*
	XTime_GetTime(&tStart);
	npu_depthconv_sw(dest, weight, W_zp, bias_values, Scale_value, 32, 128, 1, 1);
	XTime_GetTime(&tEnd);

	SW_time = 2*(tEnd - tStart);
	printf("SW Output took %lld clock cycles. %lf mseconds\r\n", SW_time, (double)SW_time/(double)1200000);
*/

	//XTime_GetTime(&tStart);
/*    npu_mem_write(DATA_BUFFER_0, 0, 2064, input_data);
    npu_mem_write(WEIGHT_BUFFER, 0, 2112, input_weight);
    //npu_depthconv_test(DEPTHCONV, DATA_BUFFER_0, DATA_BUFFER_0, 2, 2, 108, 0, 3000, 256, 255, 260, 53021371, 32, 16, 128, 1, 1, 86, 2, 43, 3, 0, 0, 1, 1);
    npu_depthconv(DEPTHCONV, DATA_BUFFER_0, DATA_BUFFER_0, 108, 0, 3000, 256, 255, 260, 53021371, 32, 16, 128, 1, 1, 43, 43, 3, 0, 0, 1, 1);
    npu_mem_read(DATA_BUFFER_0, 3000, 1968, output_data);  // Export data Length : 7872
*/

	// overflow test ************************************
	// import "npu_input_depthconv.data" to input_data address
	// import "npu_weight_bias_depthconv.data" to input_weight address
    npu_mem_write(DATA_BUFFER_0, 0, 29184, input_data); // 116736/4 = 29184
    npu_mem_write(WEIGHT_BUFFER, 0, 319, input_weight);
    npu_depthconv(DEPTHCONV, DATA_BUFFER_0, DATA_BUFFER_1, 0, 0, 0, 36, 0, 0, 477955520, -127, 0, -127, 1, 0, 114, 114, 8, 0, 0, 3, 1);
    npu_mem_read(DATA_BUFFER_1, 0, 28672, output_data);  // Export data Length : 7872

	//XTime_GetTime(&tEnd);
	//HW_time = 2*(tEnd - tStart);
	////printf(" Count per second %d\n", COUNTS_PER_SECOND);
	//printf("HW Output took %lld clock cycles. %lf mseconds\r\n", HW_time, (double)HW_time/(double)1200000);

    //5) DepthConv Group Test ---------------------------------------------
/*    char W_zp = 16;
    unsigned int Scale_value = 53021371;

	//XTime_GetTime(&tStart);
    npu_mem_write(DATA_BUFFER_0, 0, 2752, input_data);
    npu_mem_write(WEIGHT_BUFFER, 0, 2112, input_weight);
    npu_depthconv(DEPTHCONV, DATA_BUFFER_0, DATA_BUFFER_0, 2, 2, 108, 0, 3000, 256, 255, 260, 53021371, 32, 16, 128, 1, 1, 86, 2, 43, 4, 2, 1, 1, 1);
    npu_mem_read(DATA_BUFFER_0, 3000, 1312, output_data);  // Export data Length : 5248, 5238/4 = 1312
*/
    //6) Matmul Test ---------------------------------------------
    // Weight file : InitWeightBuffer_depthconv_matmul-vitis.bin
/*	XTime_GetTime(&tStart);

	XTime_GetTime(&tEnd);
	SW_time = tEnd - tStart;
	xil_printf("SW Output took %d clock cycles.\n", SW_time);*/
/*
    npu_mem_write(DATA_BUFFER_0, 0, 2064, input_data);
    npu_mem_write(WEIGHT_BUFFER, 0, 2112, input_weight);
    npu_matmul(MATMUL, DATA_BUFFER_0, DATA_BUFFER_1, 108, 0, 0, 43, 43, 2, 53021371, 32, 16, 128);
    npu_mem_read(DATA_BUFFER_1, 0, 688, output_data);  // Export data Length : 2752
*/
    //7) QLinearAdd
    // npu_qla(QLINEARADD, char buf_id, short A_base_addr, short B_base_addr, short Out_base_addr, int num_data, char A_ZP, char B_ZP, char C_ZP, int AC_Scale, int BC_Scale);

    //8) Transpose
/*
    npu_mem_write(DATA_BUFFER_0, 0, 2752, input_data);
    npu_transpose(TRANSPOSE, DATA_BUFFER_0, WEIGHT_BUFFER, 0, 0, 3, 43);
    npu_mem_read(WEIGHT_BUFFER, 0, 1536, output_data);
*/
    // Output File Generation ------------------------------------------------
/*    int end_loop;

    end_loop = 4672; //Conv3d
	//end_loop = 2336; //Conv1x1
    //end_loop = 1968; //DepthConv
    //end_loop = 1312; //DepthConv GroupMode
    //end_loop = 688; //Matmul
    //end_loop = 1536; //Transpose

	for(j=0;j<end_loop;j=j+8) {
		xil_printf("%08x%08x%08x%08x%08x%08x%08x%08x\n", output_data[j+7],output_data[j+6],output_data[j+5],output_data[j+4],
				output_data[j+3],output_data[j+2],output_data[j+1],output_data[j]);
	}

    //4. Output File Generation에 의해 Vitis Serial Terminal에 출력된 값을 선택하여 Visual Studio Editor에서 새파일을 만들고 전체선택후 "편집/고급/소문자"를 실행하여 소문자로 만든다음
	//   C:\Work\FPGA\NPU_RTL\Results\에 있는 결과파일과 비교한다.
*/

	dest = (char *)output_data;
	//char tempchar;

	//DepthConv
	/*for (j = 0; j * 128 < 28672; j++) {
		for (i = 0; i < 112; i++) {
			tempchar = dest[j * 128 + i];
			xil_printf("%d ", tempchar);
		}
		xil_printf("\n");
	}*/

	for (j = 0; j * 128 < 28672; j++) {
		for (i = 0; i < 112; i++) {
			xil_printf("%x ", dest[j * 128 + i]);
		}
		xil_printf("\n");
	}
	// copy & make outfile from xil_printf output
	// diff outfile and "NPU_outfile.txt"

    cleanup_platform();
    return 0;
}

