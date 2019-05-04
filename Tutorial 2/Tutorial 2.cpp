#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
#include <vector>
#include "ImageIO.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		ImageIO::Init(argc, argv);

		unsigned int image_width, image_height;
		std::vector<unsigned char> image_before, image_after, image_after_c, image_after_r, image_after_invert, image_after_grey;
		std::vector<unsigned char> image_before_large, image_after_large;

		ImageIO::LoadPNGImage("test.png", image_before, image_width, image_height);
		//ImageIO::LoadPNGImage("test_large.png", image_before_large, image_width, image_height);

		image_after.resize(image_before.size());
		image_after_c.resize(image_before.size());
		image_after_r.resize(image_before.size());
		image_after_invert.resize(image_before.size());
		//image_after_large.resize(image_before_large.size());
		image_after_grey.resize(image_before.size());

		ImageIO::AddWindow("Before", image_before, image_width, image_height);
		ImageIO::AddWindow("After: 1 Average Filter", image_after, image_width, image_height);
		ImageIO::AddWindow("After: 2 Convolution", image_after_c, image_width, image_height);
		ImageIO::AddWindow("After: Colour Channel Filtering for Red", image_after_r, image_width, image_height);
		ImageIO::AddWindow("After: Invert intensity value for each pixel", image_after_invert, image_width, image_height);
		//ImageIO::AddWindow("After: Invert Image Large", image_after_large, image_width, image_height);
		ImageIO::AddWindow("After: Convert input colour image into greyscale", image_after_grey, image_width, image_height);


		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f, 1.f / 9,
												1.f, 1.f / 9, 1.f,
												1.f / 9, 1.f, 1.f / 9 };

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "my_kernels2.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//device - buffers
		cl::Buffer dev_image_before(context, CL_MEM_READ_ONLY, image_before.size());
		cl::Buffer dev_image_after(context, CL_MEM_READ_WRITE, image_after.size());
		cl::Buffer dev_image_after_c(context, CL_MEM_READ_WRITE, image_after_c.size());
		cl::Buffer dev_image_after_r(context, CL_MEM_READ_WRITE, image_after_r.size());
		cl::Buffer dev_image_after_invert(context, CL_MEM_READ_WRITE, image_after_invert.size());
		// Large Image
		/*cl::Buffer dev_image_before_large(context, CL_MEM_READ_ONLY, image_before_large.size());
		cl::Buffer dev_image_after_large(context, CL_MEM_READ_WRITE, image_after_large.size());*/
		cl::Buffer dev_image_after_grey(context, CL_MEM_READ_WRITE, image_after_grey.size());
		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//Part 5 - device operations
		//5.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_before, CL_TRUE, 0, image_before.size(), &image_before[0]);
		//queue.enqueueWriteBuffer(dev_image_before_large, CL_TRUE, 0, image_before_large.size(), &image_before_large[0]);
		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "avg_filter2D");
		kernel.setArg(0, dev_image_before);
		kernel.setArg(1, dev_image_after);

		cl::Kernel kernel_c = cl::Kernel(program, "convolution2D");
		kernel_c.setArg(0, dev_image_before);
		kernel_c.setArg(1, dev_image_after_c);
		kernel_c.setArg(2, dev_convolution_mask);

		cl::Kernel kernel_r = cl::Kernel(program, "filter_r");
		kernel_r.setArg(0, dev_image_before);
		kernel_r.setArg(1, dev_image_after_r);
		kernel_r.setArg(2, dev_convolution_mask);

		cl::Kernel kernel_invert = cl::Kernel(program, "invert");
		kernel_invert.setArg(0, dev_image_before);
		kernel_invert.setArg(1, dev_image_after_invert);

		/*cl::Kernel kernel_invertLarge = cl::Kernel(program, "invert");
		kernel_invert.setArg(0, dev_image_before_large);
		kernel_invert.setArg(1, dev_image_after_large);*/

		cl::Kernel kernel_grey = cl::Kernel(program, "rgb2grey");
		kernel_grey.setArg(0, dev_image_before);
		kernel_grey.setArg(1, dev_image_after_grey);
		
	    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_c, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_r, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_invert, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);
	//	queue.enqueueNDRangeKernel(kernel_invertLarge, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_grey, cl::NullRange, cl::NDRange(image_width, image_height), cl::NullRange);


		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_after, CL_TRUE, 0, image_after.size(), &image_after[0]);
		queue.enqueueReadBuffer(dev_image_after_c, CL_TRUE, 0, image_after_c.size(), &image_after_c[0]);
		queue.enqueueReadBuffer(dev_image_after_r, CL_TRUE, 0, image_after_r.size(), &image_after_r[0]);
		queue.enqueueReadBuffer(dev_image_after_invert, CL_TRUE, 0, image_after_invert.size(), &image_after_invert[0]);
	//	queue.enqueueReadBuffer(dev_image_after_large, CL_TRUE, 0, image_after_large.size(), &image_after_large[0]);
		queue.enqueueReadBuffer(dev_image_after_grey, CL_TRUE, 0, image_after_grey.size(), &image_after_grey[0]);

		//queue.enqueueReadBuffer(dev_convolution_mask, CL_TRUE, 0, image_after_c.size(), &image_after_c[0]);

		// loop until Esc is pressed
		ImageIO::MainLoop();
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (const Exception& exc) {
		std::cerr << "ERROR: " << exc.what() << std::endl;
	}

	return 0;
}

