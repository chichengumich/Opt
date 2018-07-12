// #pragma once

// #ifndef SAFE_DELETE
// #define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
// #endif
// #ifndef SAFE_DELETE_ARRAY
// #define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
// #endif

#include "../../shared/SolverIteration.h"
#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include <fstream>

#include "Opt.h"

#include <mLibCore.h>
#include <mLibLodePNG.h>


using namespace ml;

static void loadConstraints(std::vector<std::vector<int> >& constraints, std::string filename) {
  std::ifstream in(filename, std::fstream::in);

	if(!in.good())
	{
		std::cout << "Could not open marker file " << filename << std::endl;
		assert(false);
	}

	unsigned int nMarkers;
	in >> nMarkers;
	constraints.resize(nMarkers);
	for(unsigned int m = 0; m<nMarkers; m++)
	{
		int temp;
		for (int i = 0; i < 4; ++i) {
			in >> temp;
			constraints[m].push_back(temp);
		}

	}

	in.close();
}

vec2f toVec2(float2 p) {
    return vec2f(p.x, p.y);
}

inline bool PointInTriangleLK(float x0, float y0, float w0,
float x1, float y1, float w1,
float x2, float y2, float w2,
float sx, float sy,
float *wt0, float *wt1, float *wt2) {
	float X[3], Y[3];

	X[0] = x0 - sx*w0;
	X[1] = x1 - sx*w1;
	X[2] = x2 - sx*w2;

	Y[0] = y0 - sy*w0;
	Y[1] = y1 - sy*w1;
	Y[2] = y2 - sy*w2;

	float d01 = X[0] * Y[1] - Y[0] * X[1];
	float d12 = X[1] * Y[2] - Y[1] * X[2];
	float d20 = X[2] * Y[0] - Y[2] * X[0];

	if ((d01 < 0) & (d12 < 0) & (d20 < 0)) {
		//printf("Backfacing\n");
		// backfacing
		return false;
	}

	float OneOverD = 1.f / (d01 + d12 + d20);
	d01 *= OneOverD;
	d12 *= OneOverD;
	d20 *= OneOverD;

	*wt0 = d12;
	*wt1 = d20;
	*wt2 = d01;

	return (d01 >= 0 && d12 >= 0 && d20 >= 0);
}


void rasterizeTriangle(float2 p0, float2 p1, float2 p2, vec3f c0, vec3f c1, vec3f c2, float& m_scale, ColorImageR32G32B32& m_resultColor) {
    vec2f t0 = toVec2(p0)*m_scale;
    vec2f t1 = toVec2(p1)*m_scale;
    vec2f t2 = toVec2(p2)*m_scale;


    int W = m_resultColor.getWidth();
    int H = m_resultColor.getHeight();

    vec2f minBound = math::floor(math::min(t0, math::min(t1, t2)));
    vec2f maxBound = math::ceil(math::max(t0, math::max(t1, t2)));
    for (int x = (int)minBound.x; x <= maxBound.x; ++x) {
        for (int y = (int)minBound.y; y <= maxBound.y; ++y) {
            if (x >= 0 && x < W && y >= 0 && y < H) {
                float b0, b1, b2;
                if (PointInTriangleLK(t0.x, t0.y, 1.0f,
                    t1.x, t1.y, 1.0f,
                    t2.x, t2.y, 1.0f, (float)x, (float)y, &b0, &b1, &b2)) {
                    vec3f color = c0*b0 + c1*b1 + c2*b2;
                    m_resultColor(x, y) = color;
                }

            }
        }
    }

    //bound
    //loop
    //point in trinagle
    // z-test?
}

void setConstraintImage(float alpha, 
ColorImageR32& m_image, 
ColorImageR32& m_imageMask, 
std::vector<std::vector<int>>& m_constraints,
 std::shared_ptr<OptImage>& m_constraintImage)
{
    std::vector<float2> h_constraints(m_image.getWidth()*m_image.getHeight());
    for (unsigned int y = 0; y < m_image.getHeight(); y++)
    {
        for (unsigned int x = 0; x < m_image.getWidth(); x++)
        {
            h_constraints[y*m_image.getWidth() + x] = { -1.0f, -1.0f };
        }
    }

    for (unsigned int k = 0; k < m_constraints.size(); k++)
    {
        int x = m_constraints[k][0];
        int y = m_constraints[k][1];

        if (m_imageMask(x, y) == 0)
        {
            float newX = (1.0f - alpha)*(float)x + alpha*(float)m_constraints[k][2];
            float newY = (1.0f - alpha)*(float)y + alpha*(float)m_constraints[k][3];


            h_constraints[y*m_image.getWidth() + x] = { newX, newY };
        }
    }
    m_constraintImage->update(h_constraints);
}


int main(int argc, const char * argv[]) {
	// CAT 
    std::string filename = "../data/cat512.png";

    int downsampleFactor = 1;
	bool lmOnlyFullSolve = false;
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        downsampleFactor = std::max(0,atoi(argv[2])); 
    }
    bool performanceRun = false;
    if (argc > 3) {
        if (std::string(argv[3]) == "perf") {
            performanceRun = true;
            if (atoi(argv[2]) > 0) {
                lmOnlyFullSolve = true;
            }
        } else {
            printf("Invalid third parameter: %s\n", argv[3]);
        }
    }

    // Must have a mask and constraints file in the same directory as the input image
    std::string maskFilename = filename.substr(0, filename.size() - 4) + "_mask.png";
    std::string constraintsFilename = filename.substr(0, filename.size() - 3) + "constraints";
    std::vector<std::vector<int>> constraints;
    loadConstraints(constraints, constraintsFilename);

    ColorImageR8G8B8A8 image = LodePNG::load(filename);
    const ColorImageR8G8B8A8 imageMask = LodePNG::load(maskFilename);

    ColorImageR32G32B32 imageColor(image.getWidth() / downsampleFactor, image.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < image.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < image.getWidth() / downsampleFactor; x++) {
            auto val = image(x*downsampleFactor, y*downsampleFactor);

            imageColor(x,y) = vec3f(val.x, val.y, val.z);
        }
    }

    ColorImageR32 imageR32(imageColor.getWidth(), imageColor.getHeight());
    printf("width %d, height %d\n", imageColor.getWidth(), imageColor.getHeight());
    for (unsigned int y = 0; y < imageColor.getHeight(); y++) {
        for (unsigned int x = 0; x < imageColor.getWidth(); x++) {
            imageR32(x, y) = imageColor(x, y).x;
		}
	}
    int activePixels = 0;

    ColorImageR32 imageR32Mask(imageMask.getWidth() / downsampleFactor, imageMask.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < imageMask.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < imageMask.getWidth() / downsampleFactor; x++) {
            imageR32Mask(x, y) = imageMask(x*downsampleFactor, y*downsampleFactor).x;
            if (imageMask(x*downsampleFactor, y*downsampleFactor).x == 0.0f) {
                ++activePixels;
            }
		}
	}
    printf("numActivePixels: %d\n", activePixels);
	
    for (auto& constraint : constraints) {
        for (auto& c : constraint) {
            c /= downsampleFactor;
        }
    }

    for (unsigned int y = 0; y < imageColor.getHeight(); y++)
	{
        for (unsigned int x = 0; x < imageColor.getWidth(); x++)
		{
            if (y == 0 || x == 0 || y == (imageColor.getHeight() - 1) || x == (imageColor.getWidth() - 1))
			{
				std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
				constraints.push_back(v);
			}
		}
	}

    CombinedSolverParameters params;
    params.numIter = 19;
    params.useCUDA = false;
    params.nonLinearIter = 8;
    params.linearIter = 400;
    if (performanceRun) {
        params.useCUDA = false;
        params.useOpt = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.earlyOut = true;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useOpt = false;
        params.useOptLM = true;
        params.linearIter = 500;
        if (image.getWidth() > 1024) {
            params.nonLinearIter = 100;
        }
        // TODO: Remove for < 2048x2048
    }

    CombinedSolverParameters m_combinedSolverParameters = params;

    ColorImageR32 m_image = imageR32;
    ColorImageR32G32B32 m_imageColor = imageColor;
    ColorImageR32 m_imageMask = imageR32Mask;
    std::vector<unsigned int> m_dims = { m_image.getWidth(), m_image.getHeight() };

    std::shared_ptr<OptImage> m_urshape = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
    std::shared_ptr<OptImage> m_warpField = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
    std::shared_ptr<OptImage> m_constraintImage = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
	std::shared_ptr<OptImage> m_warpAngles = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);
    std::shared_ptr<OptImage> m_mask = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);


    float weightFit = 100.0f;
    float weightReg = 0.01f;

    float m_weightFitSqrt = sqrtf(weightFit);
    float m_weightRegSqrt = sqrtf(weightReg);

    std::vector<std::vector<int>>& m_constraints = constraints;

    //reset GPU
    std::vector<float2> h_urshape(m_dims[0] * m_dims[1]);
    std::vector<float>  h_mask(m_dims[0] * m_dims[1]);
    for (unsigned int y = 0; y < m_image.getHeight(); y++)
    {
        for (unsigned int x = 0; x < m_image.getWidth(); x++)
        {
            h_urshape[y*m_image.getWidth() + x] = { (float)x, (float)y };
            h_mask[y*m_image.getWidth() + x] = (float)m_imageMask(x, y);
        }
    }

    setConstraintImage(1.0f, m_image, m_imageMask, m_constraints, m_constraintImage);

    m_urshape->update(h_urshape);
    m_warpField->update(h_urshape);
    m_mask->update(h_mask);
    cudaSafeCall(cudaMemset(m_warpAngles->data(), 0, sizeof(float)*m_image.getWidth()*m_image.getHeight()));

    //Opt C API call
    Opt_InitializationParameters initParams;
    memset(&initParams, 0, sizeof(Opt_InitializationParameters));
    initParams.verbosityLevel = 1;
    initParams.collectPerKernelTimingInfo = 1;
    initParams.doublePrecision = 0;

    Opt_State* m_optimizerState = Opt_NewState(initParams);
    Opt_Problem* m_problem = Opt_ProblemDefine(m_optimizerState, "image_warping.t", "gaussNewtonGPU");
    Opt_Plan* m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, (unsigned int*)m_dims.data());

    //solve init
    NamedParameters m_problemParams;
    m_problemParams.set("Offset",       m_warpField);
    m_problemParams.set("Angle",        m_warpAngles);
    m_problemParams.set("UrShape",      m_urshape);
    m_problemParams.set("Constraints",  m_constraintImage);
    m_problemParams.set("Mask",         m_mask);
    m_problemParams.set("w_fitSqrt",    &m_weightFitSqrt);
    m_problemParams.set("w_regSqrt",    &m_weightRegSqrt);

    //set solver parameters
    NamedParameters m_solverParams;
    m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
    m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);

    double m_finalCost = nan("");

    //solve
    for (int i = 0; i < (int)m_combinedSolverParameters.numIter; ++i) {
        std::cout << "//////////// ITERATION" << i << "  (" << "Opt" << ") ///////////////" << std::endl;
        //pre nonliner solve
        setConstraintImage((float)(i+1) / (float)m_combinedSolverParameters.numIter,
        m_image, m_imageMask, m_constraints, m_constraintImage);

        //solve
        // s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s.iterationInfo);
        NamedParameters& solverParameters = m_solverParams;
        NamedParameters& problemParameters = m_problemParams;
        NamedParameters finalProblemParameters = problemParameters;

        setAllSolverParameters(m_optimizerState, m_plan, solverParameters);

        Opt_ProblemSolve(m_optimizerState, m_plan, finalProblemParameters.data().data());
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
        
        //poseNonlinearSolve
        // postNonlinearSolve(i); //DO Nothing
        if (m_combinedSolverParameters.earlyOut) {
            break;
        }
    }

    //pose singlesolve
    //copy result to CPU
    float m_scale = 1.0f;
    ColorImageR32G32B32 m_resultColor = ColorImageR32G32B32((unsigned int)(m_image.getWidth()*m_scale), (unsigned int)(m_image.getHeight()*m_scale));
    m_resultColor.setPixels(vec3f(255.0f, 255.0f, 255.0f));

    std::vector<float2> h_warpField(m_image.getWidth()*m_image.getHeight());
    m_warpField->copyTo(h_warpField);

    // Rasterize the results
    unsigned int c = 3;
    for (unsigned int y = 0; y < m_image.getHeight(); y++)
    {
        for (unsigned int x = 0; x < m_image.getWidth(); x++)
        {
            if (y + 1 < m_image.getHeight() && x + 1 < m_image.getWidth())
            {
                if (m_imageMask(x, y) == 0)
                {
                    float2 pos00 = h_warpField[y*m_image.getWidth() + x];
                    float2 pos01 = h_warpField[y*m_image.getWidth() + (x + 1)];
                    float2 pos10 = h_warpField[(y + 1)*m_image.getWidth() + x];
                    float2 pos11 = h_warpField[(y + 1)*m_image.getWidth() + (x + 1)];

                    vec3f v00 = m_imageColor(x, y);
                    vec3f v01 = m_imageColor(x + 1, y);
                    vec3f v10 = m_imageColor(x, y + 1);
                    vec3f v11 = m_imageColor(x + 1, y + 1);

                    bool valid00 = (m_imageMask(x, y) == 0);
                    bool valid01 = (m_imageMask(x, y + 1) == 0);
                    bool valid10 = (m_imageMask(x + 1, y) == 0);
                    bool valid11 = (m_imageMask(x + 1, y + 1) == 0);

                    if (valid00 && valid01 && valid10 && valid11) {
                        rasterizeTriangle(pos00, pos01, pos10,
                            v00, v01, v10, m_scale, m_resultColor);
                        rasterizeTriangle(pos10, pos01, pos11,
                            v10, v01, v11, m_scale, m_resultColor);
                    }
                }
            }
        }
    }
   

	// CombinedSolver solver(imageR32, imageColor, imageR32Mask, constraints, params);
    // solver.solveAll();
    // ColorImageR32G32B32* res = solver.result();
    ColorImageR32G32B32* res = &m_resultColor;
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = util::boundToByte((*res)(x, y).x);
            unsigned char g = util::boundToByte((*res)(x, y).y);
            unsigned char b = util::boundToByte((*res)(x, y).z);
			out(x, y) = vec4uc(r, g, b, 255);
	
			for (unsigned int k = 0; k < constraints.size(); k++)
			{
				if (constraints[k][2] == x && constraints[k][3] == y) 
				{
                    if (imageR32Mask(constraints[k][0], constraints[k][1]) == 0)
					{
						//out(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
		
				if (constraints[k][0] == x && constraints[k][1] == y)
				{
					if (imageR32Mask(x, y) == 0)
					{
                        image(x*downsampleFactor, y*downsampleFactor) = vec4uc(255, 0, 0, 255);
					}
				}
			}
		}
	}
	LodePNG::save(out, "output.png");
	LodePNG::save(image, "inputMark.png");
    printf("Saved\n");

	return 0;
}
