

#include "fftanalysis.h"
using namespace img;
using namespace std;



void img::load_fft( const ImgProc& input, FFTImgProc& fftoutput )
{
   fftoutput.clear( input.nx(), input.ny(), input.depth() );
   for(int j=0;j<input.ny();j++)
   {
#pragma omp parallel for
      for(int i=0;i<input.nx();i++)
      {
         std::vector<float> ci;
	      std::vector< std::complex<double> > citilde;
	      input.value(i,j,ci);
	      for(size_t c=0;c<ci.size();c++)
	      {
	       std::complex<double> v(ci[c], 0.0);
	       citilde.push_back(v);
	       }
	      fftoutput.set_value(i,j,citilde);
      }
   }
}

void img::center_origin(FFTImgProc &data){

   FFTImgProc original = data;
   for(int j = 0;  j < data.ny(); j++) {
      #pragma omp parallel for         
      for(int i = 0; i < data.nx(); i++){
         int ic = data.nx()/2 + i;
         if(ic >= data.nx()) {
            ic-= data.nx();
         }
         vector<complex<double>> v0;
         data.value(i,j,v0);
         original.set_value(ic, j, v0);
      }
   }

   for(int j = 0; j < data.ny(); j++) {
      int jc = data.ny() / 2 + j;
      if (jc >= data.ny())
      {
         jc -= data.ny();
      }

      for(int  i = 0; i < data.nx(); i++) {

         vector<complex<double> > v0;
         original.value(i,j, v0);
         data.set_value(i,jc,v0);
      }
   }
}

void img::unsmoothed_psd(const ImgProc &input, ImgProc &psd) {
   FFTImgProc input_tilde;
   load_fft(input, input_tilde);
   input_tilde.fft_forward();
   cout << "center  \n";
   center_origin(input_tilde);
   psd.clear(input.nx(), input.ny(), input.depth());
   for (int j = 0; j < input.ny(); j++)
   {
      #pragma omp parallel for
      for (int i = 0; i < input.nx(); i++)
      {
         vector<float> ci;
         vector<complex<double>> citilde;
         input_tilde.value(i,j, citilde);
         ci.resize(citilde.size());
         for(size_t c = 0; c < citilde.size(); c++) {
            ci[c] = (citilde[c]* conj(citilde[c])).real();
         }
         psd.set_value(i, j, ci);

      }
   }
}

LinearWave::LinearWave(const ImgProc &init, const double dis_fac):
alpha(3.33), frame_count(0) {
   A.clear(init.nx(), init.ny(), init.depth());
   B.clear(init.nx(), init.ny(), init.depth());

}

double LinearWave::dispersion(double kx, double ky) const {

   double kmag = sqrt(kx*kx+ky*ky);
   double freq = 3.33 * sqrt(kmag);
   return freq;

}


void LinearWave::ingest(const ImgProc& I) {

   FFTImgProc Itilde;
   img::load_fft(I,Itilde);
   Itilde.fft_forward();
#pragma omp parallel for
   for(int j = 0; j < Itilde.ny(); j++) {
#pragma omp parallel for
      for(int i = 0; i < Itilde.nx(); i++) {
        

         vector<complex<double>> itilde(3);
         vector<complex<double>> a(3);
         vector<complex<double>> b(3);
         Itilde.value(i,j,itilde);
         A.value(i,j,a);
    
         B.value(i,j,b);
  

         vector<complex<double>> a_update = a;
         vector<complex<double>> b_update = b;
         complex<double> phase (0.0, frame_count*dispersion(Itilde.kx(i), Itilde.ky(j)));
         phase = exp(phase);
         double one_over_N = 1.0/(frame_count + 1);
 
         for(size_t c = 0; c < itilde.size(); c++) {
  
            a_update[c] += (itilde[c]/phase - b[c]/(phase*phase) - a[c] ) * one_over_N;
            
            b_update[c] += (itilde[c]*phase - b[c]*(phase*phase) - b[c] ) * one_over_N;
         }
        

         A.set_value(i,j,a_update);
        
         B.set_value(i, j, b_update);
      }
   }

   frame_count++;
}


void LinearWave::value(int i, int j, int n, vector<complex<double>>& amp) const{
   complex<double> phase(0.0, n*dispersion(A.kx(i), A.ky(j)));
   phase = exp(phase);

   vector<complex<double>> a;
   vector<complex<double>> b;
   
   A.value(i, j, a);
   B.value(i, j, b);
   amp.resize(a.size());
   for(size_t c = 0; c < a.size(); c++) {

      amp[c] = a[c]*phase + b[c]/phase;
   }
}

void img::extrac_image(const LinearWave &l, int frame, ImgProc &img){

   cout << "size img:  " << l.getA().nx() << endl;
   img.clear(l.getA().nx(), l.getA().ny(), l.getA().depth());
   

   FFTImgProc fftimg;
   fftimg.clear(img.nx(), img.ny(), img.depth());

   for(int j = 0; j < img.ny(); j++) {
#pragma omp parallel for
      for(int i = 0; i < img.nx(); i++)
      {
         vector<complex<double> > v;
         l.value(i,j,frame,v);
         fftimg.set_value(i,j,v);
      }
   }
   
 cout << endl;
   fftimg.fft_backward();
 //  cout << "?????\n" << endl;
#pragma omp parallel for
   for (int j = 0; j < img.ny(); j++)
   {
      for (int i = 0; i < img.nx(); i++)
      {
         vector<complex<double>> v;
         fftimg.value(i, j, v);

         vector<float> iv(v.size());
         for (size_t c = 0; c < v.size(); c++)
         {
            
            iv[c] = v[c].real();// * phase + b[c] / phase;
         }  

         img.set_value(i, j, iv);
         
      }
   }
}