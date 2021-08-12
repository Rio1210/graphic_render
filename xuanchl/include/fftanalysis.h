
#ifndef FFTANALYSIS_H
#define FFTANALYSIS_H

#include <string>
#include <cmath>
#include "fftimgproc.h"
#include "imgproc.h"
#include <vector>
#include "fftw3.h"
#include <iostream>
#include <stack>

using namespace std;

namespace img
{

void load_fft( const ImgProc& input, FFTImgProc& fftoutput );

void center_origin(FFTImgProc &data);

void unsmoothed_psd(const ImgProc& input, ImgProc& psd);
//void

class LinearWave{
    public:
        LinearWave(const ImgProc& init, const double dis_fac);
        ~LinearWave(){};

        void ingest(const ImgProc& I);
        const FFTImgProc& getA() const {return A;}

        const FFTImgProc& getB() const {return B;}

        void value(int i, int j, int n,vector <complex<double>>& amplited) const;

    protected :
        FFTImgProc A;
        FFTImgProc B;

    private:

        double alpha;
        int frame_count;
        double dispersion(double kx, double ky) const;

};

void extrac_image(const LinearWave& l, int frame, ImgProc& img);

}
#endif

// LinearWave& li;
//void LinearWave::ingest(const ImgProc& I)
