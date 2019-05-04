#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;

}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations

		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		// Wk 4 - Section 1
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device 
		// cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device) << endl; //get info
		// cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) 

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}

		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation | host - input
		//Smaller vector size with random values
		//vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		//vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

		//vector<float> A = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f }; //C++11 allows this type of initialisation
		//vector<float> B = { 0.f, 1.f, 2.f, 0.f, 1.f, 2.f, 0.f, 1.f, 2.f, 0.f };

		//Wk 3 - replacing the data set above with a larger data set
		//Larger vector size 
		size_t vec_size = 100;
		//Vector sizes for integer,double and float values
		std::vector<int>A(vec_size);
		std::vector<int>B(vec_size);

		std::vector<double>A_d(vec_size);
		std::vector <double>B_d(vec_size);

		std::vector<float>A_f(vec_size);
		std::vector<float>B_f(vec_size);

		//Add random values into the vector
		for (int i = 0; i < vec_size; i++) {
			int rnda = rand() % 1000;
			int rndb = rand() % 1000;

			A.push_back(rnda);
			B.push_back(rndb);

			//static_cast converts the data type from integer 
			A_f.push_back(static_cast<float>(rnda));
			B_f.push_back(static_cast<float>(rndb));

			A_d.push_back(static_cast<double>(rnda));
			B_d.push_back(static_cast<double>(rndb));
		}


		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size() * sizeof(int);//size in bytes
		size_t vector_size_d = A_d.size() * sizeof(double);//size in bytes for double

		//host - output
		vector<int> C(vector_elements);
		vector<float> C_f(vector_elements);
		vector<double> C_d(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);
		//device - buffer for float
		cl::Buffer buffer_Af(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_Bf(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_Cf(context, CL_MEM_READ_WRITE, vector_size);
		//device - buffer for double
		cl::Buffer buffer_Ad(context, CL_MEM_READ_WRITE, vector_size_d);
		cl::Buffer buffer_Bd(context, CL_MEM_READ_WRITE, vector_size_d);
		cl::Buffer buffer_Cd(context, CL_MEM_READ_WRITE, vector_size_d);

		//Part 5 - device operations
		//5.1 Copy arrays A and B to device memory - for integers, float and double values
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		queue.enqueueWriteBuffer(buffer_Af, CL_TRUE, 0, vector_size, &A_f[0]);
		queue.enqueueWriteBuffer(buffer_Bf, CL_TRUE, 0, vector_size, &B_f[0]);

		queue.enqueueWriteBuffer(buffer_Ad, CL_TRUE, 0, vector_size_d, &A_d[0]);
		queue.enqueueWriteBuffer(buffer_Bd, CL_TRUE, 0, vector_size_d, &B_d[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		//Wk 3 - Kernels - create another kernel variable called kernel_mult
		cl::Kernel kernel_mult = cl::Kernel(program, "mult");
		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);

		//Wk 3 - Create a new kernel called addf which will perform parallel addition of two vectors containing floating point values (float)
		cl::Kernel kernel_addf = cl::Kernel(program, "addf");
		kernel_addf.setArg(0, buffer_Af);
		kernel_addf.setArg(1, buffer_Bf);
		kernel_addf.setArg(2, buffer_Cf);

		/*cl::Kernel kernel_addd = cl::Kernel(program, "addd");
		kernel_addd.setArg(0, buffer_Ad);
		kernel_addd.setArg(1, buffer_Bd);
		kernel_addd.setArg(2, buffer_Cd);*/

		// Wk 4 - Device buffers
		cl::Kernel kernel_addID = cl::Kernel(program, "addID");
		//kernel_addID.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		kernel_addID.setArg(0, buffer_A);
		kernel_addID.setArg(1, buffer_B);
		kernel_addID.setArg(2, buffer_C);
		
	
		cl::Event prof_event_add;
		cl::Event prof_event_mult;
		cl::Event prof_event_addf;

		//Wk 3 - add the kernel launches into the queue in the right order
		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event_mult);
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event_add);
		// call a 2D kernel with data arranged into a 5x2 array
		// queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(5,2),	cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_addf, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event_addf);
		//queue.enqueueNDRangeKernel(kernel_addd, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		//Wk 3- Create an event and attach it to a queue command responsible for the kernel launch
		//queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		// Wk 4 - Length of vectors
		// Size, in bytes of each vector
		size_t global_size, local_size;
		local_size = 100;
		//Number of total work items - local 
		//global_size = ceil(n / (float)local_size)*local_size;

		queue.enqueueNDRangeKernel(kernel_addID, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
		// Wk 4 - 
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements),
			cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);
		queue.enqueueReadBuffer(buffer_Cf, CL_TRUE, 0, vector_size, &C_f[0]);

		//Wk 3 - Display the kernel execution time at the end of the program 
		std::cout << "Kernel add execution time [ns]:" << prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Kernel multiply execution time [ns]:" << prof_event_mult.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_mult.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Kernel add float execution time [ns]:" << prof_event_addf.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_addf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;


		std::cout << "Profiling Information:" << endl;
		std::cout << "\nFull Profiling Information Add:\n" << GetFullProfilingInfo(prof_event_add, ProfilingResolution::PROF_US) << endl;
		std::cout << "\nFull Profiling Information Multiply:\n" << GetFullProfilingInfo(prof_event_mult, ProfilingResolution::PROF_US) << endl;
		std::cout << "\nFull Profiling Information Add Float:\n" << GetFullProfilingInfo(prof_event_addf, ProfilingResolution::PROF_US) << endl;

		/*  cout << "\nValues within A,B,C:" << endl;
			cout << "A = " << A << endl;
			cout << "B = " << B << endl;
			cout << "C = " << C_d << endl;*/
	}

	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	cin.get();
	return 0;
}