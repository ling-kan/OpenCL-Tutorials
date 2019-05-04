//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	//printf("work item id = %d\n", id);
	C[id] = A[id] + B[id];
}

//Wk 3 - Let us try to modify the kernel code. Revert to the original input vectors consisting of 10 values only. 
//Change the kernel code so it does multiplication instead of addition (you don’t need to change the function name at this point).
//Run the code again and check if the output results are as expected.
//C[id] = A[id] *  B[id];

__kernel void mult(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

//Wk 3 - Create a new kernel called addf which will perform parallel addition of two vectors containing floating point values (float)
//Make necessary adjustments in the host code including changing the name of the executed kernel. 
//Is there any difference in performance of different devices for large vector sizes between int and float data types? 
__kernel void addf(__global const float* A, __global const float* B, __global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
	//printf("A: %f", A[id]);
}


//Wk 3 - Next, repeat the same task but this time for double precision vectors (double), perform the comparisons and note the differences. 
//Note: not all OpenCL devices support double variables: if this happens, your program will not execute correctly and report an error.
/*__kernel void addd(__global const double* A, __global const double* B, __global double* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}*/

//Wk 4 - OpenCL Kernel 
__kernel void addID(__global const int* A, __global const int* B, __global int* C) {

	// Get our global thread ID
	int id = get_global_id(0);

	// Printing out the id variable from our simple vector additional kernel
    //printf("work item id = %d\n", id);
	
	C[id] = A[id] + B[id];
	
	// Wk 4 - Note the difference between local and global ids but also work group size for different devices
	if (id == 0) { 
	//perform this part  only once i.e. for work item 0
	printf("work group size %d\n", get_local_size(0));
	}
	int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, loc_id);
	//do it for each work item
}
/*
__kernel void add2D(__global const int* A, __global const int* B, __global const int* C){ 
  	int id = get_global_id(0);

	C[id] = A[id] + B[id];

	// Wk 4 - Note the difference between local and global ids but also work group size for different devices
	if (id == 0) { 
	//perform this part  only once i.e. for work item 0
	printf("work group size %d\n", get_local_size(0));
	}
	int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, loc_id);
	//do it for each work item
}

//simple 2D identity kernel
__kernel void identity2D(__global const uchar4* A, __global uchar4* B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;
	C[id] = A[id] + B[id];
}*/