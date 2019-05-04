//a simple OpenCL kernel which copies all pixels from A to B
__kernel void identity(__global const uchar4* A, __global uchar4* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

//simple 2D identity kernel
__kernel void identity2D(__global const uchar4* A, __global uchar4* B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;
	B[id] = A[id];
}

//1D averaging filter (Radius 1 = 3 meighbours)
// Average of values at the center , and immediate neighbours to the left and right
//simple smoothing filter for noise filtering 

//Serial implementation in C
__kernel void avg_filter_serial(__global const float* a,__global float* b){
	int N;
	//handle boundary conditions
	//id = 0 and id =N-1
	for(int id = 1; id < N -1; id++)
	b[id] = (a[id -1] + a[id] + a[id+1])/3;

}

// Parallel Implementation in OpenCL
__kernel void avg_filter_parallel(__global const float* a, __global float* b){
	int id = get_global_id(0);
	//handle boundary conditions
	//id = 0 and id = N-1
	b[id]= (a[id-1] + a[id] + a[id+1])/3;
}


//2D averaging filter
__kernel void avg_filter2D(__global const uchar4* A, __global uchar4* B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;

	uint4 result = (uint4)(0);//zero all 4 components

	uint num_neighbours = 2;

	for (int i = (x-num_neighbours); i <= (x+num_neighbours); i++)
		for (int j = (y-num_neighbours); j <= (y+num_neighbours); j++) 
			result += convert_uint4(A[i + j*width]); //convert pixel values to uint4 so the sum can be larger than 255

	result /= (uint4)(9); //normalise all components (the result is a sum of 9 values) 

	B[id] = convert_uchar4(result); //convert back to uchar4 
}

//2D 3x3 convolution kernel
__kernel void convolution2D(__global const uchar4* A, __global uchar4* B, __constant float* mask) {
	int x = get_global_id(0);
	int y = get_global_id(1); 
	int width = get_global_size(0);
	int id = x + y*width;

	float4 result = (float4)(0.0f,0.0f,0.0f,0.0f);//zero all 4 components

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++)
		result += convert_float4(A[i + j*width])*(float4)(mask[i-(x-1) + (j-(y-1))*3]);//convert pixel and mask values to float4

	B[id] = convert_uchar4(result); //convert back to uchar4
}
__kernel void filter_r(__global const uchar4* A, __global uchar4* B, __constant float* mask) {
	int x = get_global_id(0);
	int y = get_global_id(1); 
	int width = get_global_size(0);
	int id = x + y*width;

	float4 result = (float4)(0.0f,0.0f,0.0f,0.0f);//zero all 4 components

	//uchar4 pixel;
	//pixel.x = 255;
	//pixel.y = 0;
	//pixel.z = 0;


	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++)
		result += convert_float4(A[i + j*width])*(float4)(mask[i-(x-1) + (j-(y-1))*3]);//convert pixel and mask values to float4
	
	//Changing (filtering) the colours except red to blank (black)	
	result.y = 0;
	result.z = 0;

	B[id] = convert_uchar4(result); //convert back to uchar4
}

__kernel void invert (__global const uchar4* inputImage, __global uchar4* outputImage){ 
	int x = get_global_id(0);
	int y = get_global_id(1); 
	int width = get_global_size(0);
	int id = x + y*width;

	//access pixel value and store it in a new var
	uchar4 inputPixel = inputImage[id];
	//calculate the inverted value and store it in a new var
	uchar4 invertedPixel;
	invertedPixel.x = 255-inputPixel.x;
	invertedPixel.y = 255-inputPixel.y;
	invertedPixel.z = 255-inputPixel.z;
	invertedPixel.w = inputPixel.w;
	//append the inverted value into image B
	outputImage[id] = invertedPixel;
}

__kernel void rgb2grey (__global const uchar4* inputImage, __global uchar4* outputImage){ 
	int x = get_global_id(0);
	int y = get_global_id(1); 
	int width = get_global_size(0);
	int id = x + y*width;

	//get the pixel value and store in a new vara
	uchar4 inputPixel = inputImage[id];

	//Convert the colour to grey scale -  using the  formulay=02126r + 07152g + 0.0722b;
	uchar4 greyImage = 0.2126f * inputPixel.x + 0.7152f * inputPixel.y + 0.0722f * inputPixel.z;
	//Put the grey image to output for each pixel value
	outputImage[id] = greyImage;
}


	